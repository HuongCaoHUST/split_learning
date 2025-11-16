import math
import torch
import time
import warnings
import numpy as np
import os
import uuid
import pickle
import pika
import pandas as pd
from copy import copy
from pathlib import Path
from typing import Dict, Optional, Union
from torch import distributed as dist
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.classify import ClassificationTrainer
from typing import Optional
from ultralytics.utils import RANK
from engine.model import Split_Learning_DetectionModel, Split_Learning_SegmentationModel, Split_Learning_ClassificationModel
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics import __version__
from ultralytics.utils.checks import check_amp, check_imgsz
from ultralytics.data.utils import check_cls_dataset
from ultralytics.data import build_dataloader
from ultralytics.utils.plotting import plot_results
from engine.data import check_det_dataset
from copy import deepcopy
from datetime import datetime
from src import Utils
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)
import threading

class Split_Learning_DetectionTrainer(DetectionTrainer):
    def __init__(self, overrides, client_id=None, layer_id=None, num_client=None, cut_layer=None, address=None, username=None, password=None, load_partial_model=False, FedAvg=False):
        self.client_id = client_id
        self.layer_id = layer_id
        self.num_client = num_client
        self.client_ids = []
        self.cut_layer = cut_layer
        self.address = address
        self.username = username
        self.password = password
        self.load_partial_model = load_partial_model
        Utils.init_csv('./log/latency.csv', ['batch_id', 'start', 'end', 'latency'])
        Utils.init_csv('./log/com_cost.csv', ['batch_id', 'label/tensor', 'size'])
        self.status_train = False
        self.count_batch = 0
        self.channel = Utils.connect_rabbitmq(self.address, self.username, self.password)
        # self.channel.basic_consume(queue=f'gradient_queue_{self.client_id}', on_message_callback=self.on_request)
        if self.layer_id == 1:
            self.condition = threading.Condition()
            self.channel_thread = Utils.connect_rabbitmq(self.address, self.username, self.password)
            self.backward_flag = False
            self.num_forward = 0

        self.validate_intermediate = True
        super().__init__(overrides=overrides)
        if FedAvg:
            self.epochs = overrides.get("epochs", 100)
    
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank, drop_last=True)  # return dataloader; Drop_last for split_learning
    
    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        model = Split_Learning_DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1,
                            layer_id=getattr(self, 'layer_id', None),
                            client_id=getattr(self, 'client_id', None),
                            num_client=getattr(self, 'num_client', None),
                            cut_layer=getattr(self, 'cut_layer', None),
                            address=getattr(self, 'address', None),
                            username=getattr(self, 'username', None),
                            password=getattr(self, 'password', None),
                            load_partial_model=getattr(self, 'load_partial_model', False))
        if weights:
            model.load(weights)
        return model
    
    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        if self.layer_id == 1:
            return None
        elif self.layer_id == 2: 
            return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
                "Epoch",
                "GPU_mem",
                *self.loss_names,
                "Instances",
                "Size",
            )
        else:
            return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
            )
    
    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        if self.layer_id == 1:
            self.train_loader = self.get_dataloader(
                self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
            )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots and self.layer_id == 1:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        if self.layer_id == 1:
            iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
            self.optimizer = self.build_optimizer(
                model=self.model,
                name=self.args.optimizer,
                lr=self.args.lr0,
                momentum=self.args.momentum,
                decay=weight_decay,
                iterations=iterations,
            )
        else:
            self.optimizer = self.build_optimizer(
                model=self.model,
                name=self.args.optimizer,
                lr=self.args.lr0,
                momentum=self.args.momentum,
                decay=weight_decay,
            )

        # Tensor IDS get
        if self.layer_id == 1:
            self.tensor_send_ids = self.get_tensor_send_id(self.cut_layer)
        elif self.layer_id == 2:
            self.cut_layer_ids = []
            self.tensor_send_ids = []

        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train the model with the specified world size."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        self.model.channel = self.channel
        if self.layer_id == 1:
            nb = len(self.train_loader)  # number of batches
            nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        else:
            nb = self.wait_for_number_batch_client_id()
            print("Self.tensor_send_ids: ", self.tensor_send_ids)
            print("Seld.client_ids: ", self.client_ids)
            print("Seld.cut_layer_ids: ", self.cut_layer_ids)
            print("Sum number batch: ", nb)
            self.model.client_ids = self.client_ids
            self.model.cut_layer_ids = self.cut_layer_ids
            self.model.tensor_send_ids = self.tensor_send_ids
            nw = 1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        if self.layer_id == 1:
            LOGGER.info(
                f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
            )
        else:
            LOGGER.info(
                f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
            )

        # Set training flag
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'is_training'):
            self.model.module.is_training = True
        elif hasattr(self.model, 'is_training'):
            self.model.is_training = True
        else:
            LOGGER.warning(
                "Model does not have 'is_training' attribute. Ensure model is an instance of DetectionModel.")
            
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        if self.layer_id == 1:
            LOGGER.info(f"START TRAINING IN CLIENT 1")
            while True:
                start_epoch_time = time.time()
                self.epoch = epoch
                self.run_callbacks("on_train_epoch_start")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                    self.scheduler.step()

                self._model_train()
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)
                # Update dataloader attributes (optional)
                if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()

                if RANK in {-1, 0}:
                    # LOGGER.info(self.progress_string())
                    pbar = TQDM(enumerate(self.train_loader), total=nb)
                self.tloss = None

                # Send number_batch to RabbitMQ
                if epoch == 0:
                    success = self.send_number_batch_client_id(nb, self.client_id, self.cut_layer, self.tensor_send_ids)
                    if not success:
                        print(f"Không thể gửi number_batch tới queue.")
                if self.model.is_training == True:
                    self.start_thread()
                #Training loop   
                for i, batch in pbar:
                    self.run_callbacks("on_train_batch_start")
                    # Warmup
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                    # Forward
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        label_data = {
                            "batch_idx": batch["batch_idx"],
                            "bboxes": batch["bboxes"],
                            "cls": batch["cls"]
                        }
                        if self.layer_id == 1:
                            data_id = uuid.uuid4()
                            self.model.batch_id = data_id
                            success = self.send_label(data_id, label_data)
                            start_batch_time = time.time()
                            if not success:
                                print(f"Không thể gửi batch {i} tới label_queue.")

                        # Forward in task
                        preds = self.model(batch["img"])

                        Utils.log_to_csv('./log/latency.csv', {
                            'batch_id': data_id,
                            'start': start_batch_time,
                        })
                    self.num_forward += 1
                    print ("BACKWARD FLAG: ", self.backward_flag)
                    print(f"FORWARD count: {self.num_forward}/{nb}")
                    print(f"BACKWARD count: {self.count_batch}/{nb}")
                    if self.backward_flag and self.num_forward < nb:
                        success_grad, gradient_dict = self.wait_gradient()
                        if not success_grad:
                            print("Không thấy Gradient.")
                            return
                        
                        tensor_list = [self.model.data_store[t_id] for t_id in gradient_dict.keys()]
                        grad_list = [gradient_dict[t_id] for t_id in gradient_dict.keys()]

                        torch.autograd.backward(tensor_list, grad_list)
                        self.count_batch += 1

                        for g in self.optimizer.param_groups:
                            for p in g['params']:
                                if p.grad is not None:
                                    p.grad.data = p.grad.data.float()  # đảm bảo FP32
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    elif self.backward_flag or self.num_forward == nb:
                        print("FINAL BATCH")
                        if self.count_batch >= nb:
                            self.count_batch = nb - 1
                        while self.count_batch < nb:
                            success_grad, gradient_dict = self.wait_gradient()
                            if not success_grad:
                                print("Không thấy Gradient.")
                                return
                            
                            tensor_list = [self.model.data_store[t_id] for t_id in gradient_dict.keys()]
                            grad_list = [gradient_dict[t_id] for t_id in gradient_dict.keys()]

                            torch.autograd.backward(tensor_list, grad_list)
                            self.count_batch += 1
                            print(f"BACKWARD count: {self.count_batch}/{nb}")

                            for g in self.optimizer.param_groups:
                                for p in g['params']:
                                    if p.grad is not None:
                                        p.grad.data = p.grad.data.float()  # đảm bảo FP32
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break


                    # success_grad, gradient_dict = self.wait_gradient()

                    # if not success_grad:
                    #     print("Không thấy Gradient.")
                    #     return
                    
                    # tensor_list = [self.model.data_store[t_id] for t_id in gradient_dict.keys()]
                    # grad_list = [gradient_dict[t_id] for t_id in gradient_dict.keys()]

                    # torch.autograd.backward(tensor_list, grad_list)
                    # self.count_batch += 1

                    # # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                    # if ni - last_opt_step >= self.accumulate:
                    #     self.optimizer_step()
                    #     last_opt_step = ni

                    # # Timed stopping
                    # if self.args.time:
                    #     self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                    #     if RANK != -1:  # if DDP training
                    #         broadcast_list = [self.stop if RANK == 0 else None]
                    #         dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                    #         self.stop = broadcast_list[0]
                    #     if self.stop:  # training time exceeded
                    #         break

                    # Log
                    if RANK in {-1, 0}:
                        pbar.set_description(f"{epoch + 1}/{self.epochs}")
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)
                            
                    self.run_callbacks("on_train_batch_end")
                self.wait_all_backward(expected_num=nb)

                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
                self.run_callbacks("on_train_epoch_end")
                self.num_forward = 0
                if RANK in {-1, 0}:
                    final_epoch = epoch + 1 >= self.epochs
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                    # Stopper
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save model
                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                # Scheduler
                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.run_callbacks("on_fit_epoch_end")
                self._clear_memory(0.5)  # clear if memory utilization > 50%

                # Early Stopping
                if RANK != -1:  # if DDP training
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                    self.stop = broadcast_list[0]
                if self.stop:
                    break  # must break all DDP ranks
                epoch += 1
        else:
            LOGGER.info(f"START TRAINING IN CLIENT 2")
            while True:
                self.epoch = epoch
                self.run_callbacks("on_train_epoch_start")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                    self.scheduler.step()

                self._model_train()
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)

                fake_batches = [None] * nb
                pbar = enumerate(fake_batches)

                if RANK in {-1, 0}:
                    LOGGER.info(self.progress_string())
                    pbar = TQDM(enumerate(fake_batches), total=nb)
                self.tloss = None

                #Training loop
                for i, batch in pbar:
                    start_batch_forward_time = time.time()
                    self.run_callbacks("on_train_batch_start")
                    # Warmup
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                    batch = self.wait_for_batch()

                    # print("Batch received" , batch["batch_idx"])
                    # Forward
                    with autocast(self.amp):
                        # batch = self.preprocess_batch(batch)
                        batch["img"] = torch.zeros((1, 3, 640, 640)).to(self.device)
                        loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= world_size
                        self.tloss = (
                            (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                        )

                    # Backward
                    self.scaler.scale(self.loss).backward()

                    if self.layer_id == 2:
                        if hasattr(self.model, 'saved_tensor'):
                            gradient_store = {}
                            for tensor_id, tensor in self.model.saved_tensor.items():
                                if tensor.grad is not None:
                                    print(f"Gradient shape của tensor {tensor_id}: {tensor.grad.shape}")
                                    gradient_store[tensor_id] = tensor.grad
                                else:
                                    print(f"Gradient của tensor {tensor_id} là None")
                            
                            # Send gradients to gradient_queue
                            if gradient_store:
                                data_id = self.model.input_data_id
                                success = self.send_gradient(data_id, gradient_store)
                                if not success:
                                    print(f"Không thể gửi Gradients {i} tới Gradient_queue.")
                        
                        if hasattr(self.model, 'saved_data_store'):
                            for tensor_id, tensor in self.model.saved_data_store.items():
                                if tensor.grad is not None:
                                    print(f"Gradient shape của tensor {tensor_id} (data_store): {tensor.grad.shape}")
                                else:
                                    print(f"Gradient của tensor {tensor_id} (data_store) là None")
                    
                    # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni

                        # Timed stopping
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break

                    # Log
                    if RANK in {-1, 0}:
                        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                                *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                                batch["cls"].shape[0],  # batch size, i.e. 8
                                batch["img"].shape[-1],  # imgsz, i.e 640
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        # if self.args.plots and ni in self.plot_idx:
                        #     self.plot_training_samples(batch, ni)

                    self.run_callbacks("on_train_batch_end")

                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
                self.run_callbacks("on_train_epoch_end")
                if RANK in {-1, 0}:
                    final_epoch = epoch + 1 >= self.epochs
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                    # Validation
                    if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                        self._clear_memory(threshold=0.5)  # prevent VRAM spike
                        self.metrics, self.fitness = self.validate()
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save model
                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                # Scheduler
                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.run_callbacks("on_fit_epoch_end")
                self._clear_memory(0.5)  # clear if memory utilization > 50%

                # Early Stopping
                if RANK != -1:  # if DDP training
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                    self.stop = broadcast_list[0]
                if self.stop:
                    break  # must break all DDP ranks
                epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            # self.final_eval()
            if self.args.plots and self.layer_id != 1:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def start_thread(self):
        """START THREADING"""
        thread = threading.Thread(target=self.check_gradient_queue, daemon=True)
        thread.start()

    def stop_thread(self):
        """STOP THREADING"""
        self.model.is_training = False
        print(f"Thread đã dừng.")

    def send_label(self, data_id, labels):
        queue_name = f'label_queue'
        self.channel.queue_declare(queue_name, durable=False)
        # print("Label IDX: ", labels['cls'])
        message = pickle.dumps(
            {"data_id": data_id,
            "label": labels}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )

        Utils.log_to_csv('./log/com_cost.csv', {
                            'batch_id': data_id,
                            'label/tensor': "label",
                            'size': len(message)
                        })

        print(f"Batch {data_id} đã được gửi tới {queue_name}, Kích thước: {len(message)} bytes")
        return True
    
    def send_number_batch_client_id(self, nb = None, client_id = None, client_cut_layer = None, tensor_send_ids = None):
        queue_name = f'number_batch_queue'
        self.channel.queue_declare(queue_name, durable=False)

        message = pickle.dumps(
            {"nb": nb,
             "client_id": client_id,
             "client_cut_layer": client_cut_layer,
             "tensor_send_ids": tensor_send_ids}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )
        print(f"Number batch đã được gửi tới {queue_name}")
        return True
    
    def wait_for_number_batch_client_id(self):
        expected_messages = self.num_client[0]
        total_nb = 0
        received = 0
        while received < expected_messages:
            queue_name = f'number_batch_queue'
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                print("Received data:", received_data)
                nb = received_data["nb"]
                client_id = received_data["client_id"]
                client_cut_layer = received_data["client_cut_layer"]
                tensor_send_ids = received_data["tensor_send_ids"]
                if nb is not None and client_id is not None:
                    total_nb += nb
                    self.client_ids.append(client_id) 
                    self.cut_layer_ids.append(client_cut_layer) 
                    self.tensor_send_ids.append(tensor_send_ids) 
                    received += 1
            else:
                time.sleep(0.5)
        # self.channel.queue_delete(queue=queue_name)
        return total_nb
    
    def wait_for_batch(self):
        while True:
            queue_name = f'label_queue'
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                data_id = received_data["data_id"]
                data = received_data["label"]
                print(f"Received BATCH with data_id: {data_id}")
                return data
            else:
                # print("No data received yet, waiting...")
                time.sleep(1)

    def send_gradient(self, data_id, gradients):
        # queue_name = f'gradient_queue_{self.layer_id - 1}'
        client_id = data_id.split("_")[0]
        queue_name = f'gradient_queue_{client_id}'

        self.channel.queue_declare(queue_name, durable=False)

        message = pickle.dumps(
            {"data_id": data_id,
            "gadients": gradients}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )
        Utils.log_to_csv('./log/com_cost.csv', {
                            'batch_id': data_id,
                            'label/tensor': "gradient",
                            'size': len(message)
                        })

        print(f"Gradients {data_id} đã được gửi tới {queue_name}, Kích thước: {len(message)} bytes")
        return True

    def wait_gradient(self):
        """
        Wait for gradient data from the gradient_queue.

        Returns:
            tuple: (success_flag, grad4, grad6, grad10)
        """
        while True:
            queue_name = f'gradient_queue_{self.layer_id}'
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame and body:
                try:
                    received_data = pickle.loads(body)
                    data_id = received_data.get('data_id')
                    gradient_store = received_data.get('gadients', {})

                    if not isinstance(gradient_store, dict):
                        raise ValueError("Received 'gadients' is not a valid dictionary")

                    gradient_dict = {}

                    for tensor_id in self.tensor_send_ids:
                        grad = gradient_store.get(tensor_id)
                        if grad is None:
                            raise ValueError(f"Missing gradient for tensor_id {tensor_id}")
                        if not isinstance(grad, torch.Tensor):
                            raise ValueError(f"Gradient for tensor_id {tensor_id} is not a valid tensor")
                        print(f"Received gradient for tensor_id {tensor_id}, shape: {grad.shape}")
                        gradient_dict[tensor_id] = grad

                    return True, gradient_dict

                except (pickle.UnpicklingError, ValueError) as e:
                    print(f"Error processing gradient queue data: {e}")
                    time.sleep(0.5)
            else:
                # print("No gradient data received yet, waiting...")
                time.sleep(0.5)

    def check_gradient(self):
        thread_channel = self.channel_thread
        print("threading")
        queue_name = f'gradient_queue_{self.client_id}'
        while self.model.is_training:
            try:
                if thread_channel is not None and thread_channel.is_open:
                    method_frame, header_frame, body = thread_channel.basic_get(queue=queue_name, auto_ack=True)
                    if method_frame and body:
                        start_batch_backward_time = time.time()
                        received_data = pickle.loads(body)
                        data_id = received_data.get('data_id')
                        print("\nDATA_ID backward: ", data_id)
                        gradient_store = received_data.get('gadients', {})
                        if not isinstance(gradient_store, dict):
                            raise ValueError("Received 'gadients' is not a valid dictionary")
                        
                        gradient_dict = {}

                        for tensor_id in self.tensor_send_ids:
                            grad = gradient_store.get(tensor_id)
                            if grad is None:
                                raise ValueError(f"Missing gradient for tensor_id {tensor_id}")
                            if not isinstance(grad, torch.Tensor):
                                raise ValueError(f"Gradient for tensor_id {tensor_id} is not a valid tensor")
                            print(f"Received gradient for tensor_id {tensor_id}, shape: {grad.shape}")
                            gradient_dict[tensor_id] = grad

                        # Backward
                        tensor_list = [self.model.data_store[t_id] for t_id in gradient_dict.keys()]
                        grad_list = [gradient_dict[t_id] for t_id in gradient_dict.keys()]
                        torch.autograd.backward(tensor_list, grad_list)
                        self.count_batch += 1

                        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                        # self.optimizer_step()
                        for g in self.optimizer.param_groups:
                            for p in g['params']:
                                if p.grad is not None:
                                    p.grad.data = p.grad.data.float()  # đảm bảo FP32
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break

                        # Log
                        end_batch_time = time.time()
                        Utils.log_to_csv('./log/latency.csv', {
                            'batch_id': data_id.split("_")[1],
                            'start': "Null",
                            'end': end_batch_time
                        })
                else:
                    print("Thread channel is None or closed")
            except Exception as e:
                print("Error in check_gradient thread:", e)
                break
            time.sleep(2)

    def check_gradient_queue(self):
        queue_name = f'gradient_queue_{self.client_id}'
        try:
            while True:
                q = self.channel_thread.queue_declare(queue=queue_name, passive=True)
                message_count = q.method.message_count

                if message_count > 0:
                    print(f"[{queue_name}] Có {message_count} bản tin trong queue.")
                    self.backward_flag = True
                else:
                    self.backward_flag = False

                time.sleep(1)
        except pika.exceptions.ChannelClosedByBroker:
            print(f"⚠️ Queue '{queue_name}' không tồn tại trên RabbitMQ server.")
        except Exception as e:
            print(f"Lỗi khi kiểm tra queue: {e}")

    def backward_function(self):
        print("BACKWARDING FUNCTION NÈ!!!!!!!!!!!!!!!")
        queue_name = f'gradient_queue_{self.client_id}'
        while self.model.is_training:
            try:
                if self.channel_thread is not None and self.channel_thread.is_open:
                    method_frame, header_frame, body = self.channel_thread.basic_get(queue=queue_name, auto_ack=True)
                    if method_frame and body:
                        start_batch_backward_time = time.time()
                        received_data = pickle.loads(body)
                        data_id = received_data.get('data_id')
                        print("\nDATA_ID backward: ", data_id)
                        gradient_store = received_data.get('gadients', {})
                        if not isinstance(gradient_store, dict):
                            raise ValueError("Received 'gadients' is not a valid dictionary")
                        
                        gradient_dict = {}

                        for tensor_id in self.tensor_send_ids:
                            grad = gradient_store.get(tensor_id)
                            if grad is None:
                                raise ValueError(f"Missing gradient for tensor_id {tensor_id}")
                            if not isinstance(grad, torch.Tensor):
                                raise ValueError(f"Gradient for tensor_id {tensor_id} is not a valid tensor")
                            print(f"Received gradient for tensor_id {tensor_id}, shape: {grad.shape}")
                            gradient_dict[tensor_id] = grad

                        # Backward
                        tensor_list = [self.model.data_store[t_id] for t_id in gradient_dict.keys()]
                        grad_list = [gradient_dict[t_id] for t_id in gradient_dict.keys()]
                        torch.autograd.backward(tensor_list, grad_list)
                        self.count_batch += 1

                        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                        # self.optimizer_step()
                        for g in self.optimizer.param_groups:
                            for p in g['params']:
                                if p.grad is not None:
                                    p.grad.data = p.grad.data.float()  # đảm bảo FP32
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break

                        # Log
                        end_batch_time = time.time()
                        Utils.log_to_csv('./log/latency.csv', {
                            'batch_id': data_id.split("_")[1],
                            'start': "Null",
                            'end': end_batch_time
                        })
                else:
                    print("Thread channel is None or closed")
            except Exception as e:
                print("Error in check_gradient thread:", e)
                break
            time.sleep(2)

    def wait_all_backward(self, expected_num):
        with self.condition:
            while self.count_batch < expected_num:
                # print(f"Waiting... Current: {self.count_batch}/{expected_num}")
                self.condition.wait(timeout=1)

            print("Enough gradients received. Continue training.")
            self.count_batch = 0

    def wait_gradient(self):
        """
        Wait for gradient data from the gradient_queue.

        Returns:
            tuple: (success_flag, grad4, grad6, grad10)
        """
        tensor_send_ids = self.get_tensor_send_id(self.cut_layer)
        while True:
            queue_name = f'gradient_queue_{self.client_id}'
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame and body:
                try:
                    received_data = pickle.loads(body)
                    data_id = received_data.get('data_id')
                    gradient_store = received_data.get('gadients', {})

                    if not isinstance(gradient_store, dict):
                        raise ValueError("Received 'gadients' is not a valid dictionary")

                    gradient_dict = {}

                    for tensor_id in tensor_send_ids:
                        grad = gradient_store.get(tensor_id)
                        if grad is None:
                            raise ValueError(f"Missing gradient for tensor_id {tensor_id}")
                        if not isinstance(grad, torch.Tensor):
                            raise ValueError(f"Gradient for tensor_id {tensor_id} is not a valid tensor")
                        print(f"Received gradient for tensor_id {tensor_id}, shape: {grad.shape}")
                        gradient_dict[tensor_id] = grad

                    return True, gradient_dict

                except (pickle.UnpicklingError, ValueError) as e:
                    print(f"Error processing gradient queue data: {e}")
                    time.sleep(1)
            else:
                # print("No gradient data received yet, waiting...")
                time.sleep(1)


    def get_tensor_send_id (self, cut_layer):
        # tensor_send_id = []
        # mf_values = []
        # layer_indices = []
        # for idx, m in enumerate(self.model.model):
        #     f = m.f
        #     if f != -1:
        #         if isinstance(f, int):
        #             f = [f]
        #         for fi in f:
        #             if fi != -1:
        #                 layer_indices.append(idx)
        #                 mf_values.append(fi)
        # mf_values_sorted = sorted(mf_values)

        # for value in mf_values_sorted:
        #     if value < cut_layer:
        #         tensor_send_id.append(value)

        # indices_to_mf = dict(zip(layer_indices, mf_values))
        # for idx, val in indices_to_mf.items():
        #     if idx <=cut_layer:
        #         tensor_send_id.remove(val)

        # tensor_send_id.append(cut_layer)
        # print ("SEND tensor id: ", tensor_send_id)
        # return tensor_send_id
    
        if cut_layer <=3:
            return [cut_layer]
        elif cut_layer == 4:
            return [cut_layer]
        elif cut_layer == 5:
            return [4, cut_layer]
        elif cut_layer == 6:
            return [4, cut_layer]
        elif cut_layer == 7:
            return [4, 6, cut_layer]
        elif cut_layer == 8:
            return [4, 6, cut_layer]
        elif cut_layer == 9:
            return [4, 6, cut_layer]
        elif cut_layer == 10:
            return [4, 6, cut_layer]
        elif cut_layer == 11:
            return [4, 6, 10, cut_layer]
        elif cut_layer == 12:
            return [4, 10, cut_layer]
        elif cut_layer == 13:
            return [4, 10, cut_layer]
        elif cut_layer == 14:
            return [4, 10, 13, cut_layer]
        elif cut_layer == 15:
            return [10, 13, cut_layer]
        elif cut_layer == 16:
            return [10, 13, cut_layer]
        elif cut_layer == 17:
            return [10, 13, 16, cut_layer]
        elif cut_layer == 18:
            return [10, 16, cut_layer]
        elif cut_layer == 19:
            return [10, 16, cut_layer]
        elif cut_layer == 20:
            return [10, 16, 19, cut_layer]
        elif cut_layer == 21:
            return [16, 19, cut_layer]
        elif cut_layer == 22:
            return [16, 19, cut_layer]
        elif cut_layer == 23:
            return [16, 19, 22]

    def send_epoch_intermediate(self, epoch_intermediate_path = None):
        queue_name = f'Server_queue'
        self.channel.queue_declare(queue_name, durable=False)
        epoch_intermediate_path = str(epoch_intermediate_path).replace("F:\\Do_an\\split_learning", "/app").replace("\\", "/")
        message = pickle.dumps(
            {"action": "VAL_INTER",
             "client_id": self.client_id,
             "layer_id": self.layer_id,
             "epoch": self.epoch,
             "epoch_intermediate": epoch_intermediate_path}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )
        print(f"Epoch intermediate path đã được gửi tới {queue_name}")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        if self.layer_id == 1:
            torch.save(
                {
                    "epoch": self.epoch,
                    "best_fitness": self.best_fitness,
                    "model": None,  # resume and final checkpoints derive from EMA
                    "ema": deepcopy(self.ema.ema).half(),
                    "updates": self.ema.updates,
                    "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                    "train_args": vars(self.args),  # save as dict
                    "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                    # "train_results": self.read_results_csv(),
                    "date": datetime.now().isoformat(),
                    "version": __version__,
                    "license": "AGPL-3.0 (https://ultralytics.com/license)",
                    "docs": "https://docs.ultralytics.com",
                },
                buffer,
            )
        elif self.layer_id == 2:
            torch.save(
                {
                    "epoch": self.epoch,
                    "best_fitness": self.best_fitness,
                    "model": None,  # resume and final checkpoints derive from EMA
                    "ema": deepcopy(self.ema.ema).half(),
                    "updates": self.ema.updates,
                    "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                    "train_args": vars(self.args),  # save as dict
                    "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                    "train_results": self.read_results_csv(),
                    "date": datetime.now().isoformat(),
                    "version": __version__,
                    "license": "AGPL-3.0 (https://ultralytics.com/license)",
                    "docs": "https://docs.ultralytics.com",
                },
                buffer,
            )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0) and self.validate_intermediate == True:
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
            model_path = self.wdir / f"epoch{self.epoch}.pt"
            self.send_epoch_intermediate(model_path)
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint

    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data, self.layer_id)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data
    
    def final_eval(self):
        """Perform final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    if self.layer_id != 1:
                        LOGGER.info(f"\nValidating {f}...")
                        self.validator.args.plots = self.args.plots
                        self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        # print("[CHECK] Resuming training...")
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")

        if self.layer_id == 1:
            nb = len(self.train_loader)
            print("[CHECK]Number of batches:", nb)
            success = self.send_number_batch_client_id(nb, self.client_id, self.cut_layer, self.tensor_send_ids)
            if not success:
                print(f"Không thể gửi number_batch tới queue.")

        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

class Split_Learning_SegmentationTrainer(Split_Learning_DetectionTrainer):
    def __init__(self, overrides, client_id=None, layer_id=None, num_client=None, cut_layer=None, address=None, 
                username=None, password=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(
            overrides=overrides,
            client_id=client_id,
            layer_id=layer_id,
            num_client=num_client,
            cut_layer=cut_layer,
            address=address,
            username=username,
            password=password,
        )

    def get_model(
        self, cfg: Optional[Union[Dict, str]] = None, weights: Optional[Union[str, Path]] = None, verbose: bool = True
    ):
        model = Split_Learning_SegmentationModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            layer_id=getattr(self, 'layer_id', None),
            client_id=getattr(self, 'client_id', None),
            num_client=getattr(self, 'num_client', None),
            cut_layer=getattr(self, 'cut_layer', None),
            address=getattr(self, 'address', None),
            username=getattr(self, 'username', None),
            password=getattr(self, 'password', None)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a SegmentationValidator cho split learning segmentation."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plot metrics segmentation."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)

class Split_Learning_ClassificationTrainer(ClassificationTrainer):
    def __init__(self, overrides, client_id=None, layer_id=None, num_client=None, cut_layer=None, address=None, username=None, password=None):
        self.client_id = client_id
        self.layer_id = layer_id
        self.num_client = num_client
        self.client_ids = []
        self.cut_layer = cut_layer
        self.address = address
        self.username = username
        self.password = password
        Utils.init_csv('./log/log_time.csv', ['layer_id', 'client_id', 'epoch', 'forward/backward/end_epoch', 'duration'])
        self.status_train = False
        self.count_batch = 0
        if self.layer_id == 1:
            self.condition = threading.Condition()

        self.validate_intermediate = True
        super().__init__(overrides=overrides)
    
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank, drop_last=True)  # return dataloader; Drop_last for split_learning
    
    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        model = Split_Learning_ClassificationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1,
                            layer_id=getattr(self, 'layer_id', None),
                            client_id=getattr(self, 'client_id', None),
                            num_client=getattr(self, 'num_client', None),
                            cut_layer=getattr(self, 'cut_layer', None),
                            address=getattr(self, 'address', None),
                            username=getattr(self, 'username', None),
                            password=getattr(self, 'password', None))
        if weights:
            model.load(weights)
        return model
    
    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        if self.layer_id == 1:
            return None
        elif self.layer_id == 2: 
            return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
                "Epoch",
                "GPU_mem",
                *self.loss_names,
                "Instances",
                "Size",
            )
        else:
            return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
            )
    
    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        if self.layer_id == 1:
            self.train_loader = self.get_dataloader(
                self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
            )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots and self.layer_id == 1:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        if self.layer_id == 1:
            iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
            self.optimizer = self.build_optimizer(
                model=self.model,
                name=self.args.optimizer,
                lr=self.args.lr0,
                momentum=self.args.momentum,
                decay=weight_decay,
                iterations=iterations,
            )
        else:
            self.optimizer = self.build_optimizer(
                model=self.model,
                name=self.args.optimizer,
                lr=self.args.lr0,
                momentum=self.args.momentum,
                decay=weight_decay,
            )

        # Tensor IDS get
        if self.layer_id == 1:
            self.tensor_send_ids = self.get_tensor_send_id(self.cut_layer)
        elif self.layer_id == 2:
            self.cut_layer_ids = []
            self.tensor_send_ids = []

        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train the model with the specified world size."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        self.model.channel = self.channel
        if self.layer_id == 1:
            nb = len(self.train_loader)  # number of batches
            nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        else:
            nb = self.wait_for_number_batch_client_id()
            print("Self.tensor_send_ids: ", self.tensor_send_ids)
            print("Seld.client_ids: ", self.client_ids)
            print("Seld.cut_layer_ids: ", self.cut_layer_ids)
            print("Sum number batch: ", nb)
            self.model.client_ids = self.client_ids
            self.model.cut_layer_ids = self.cut_layer_ids
            self.model.tensor_send_ids = self.tensor_send_ids
            nw = 1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        if self.layer_id == 1:
            LOGGER.info(
                f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
            )
        else:
            LOGGER.info(
                f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
            )

        # Set training flag
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'is_training'):
            self.model.module.is_training = True
        elif hasattr(self.model, 'is_training'):
            self.model.is_training = True
        else:
            LOGGER.warning(
                "Model does not have 'is_training' attribute. Ensure model is an instance of DetectionModel.")
            
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        if self.layer_id == 1:
            LOGGER.info(f"START TRAINING IN CLIENT 1")
            while True:
                start_epoch_time = time.time()
                self.epoch = epoch
                self.run_callbacks("on_train_epoch_start")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                    self.scheduler.step()

                self._model_train()
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)
                # Update dataloader attributes (optional)
                if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()

                if RANK in {-1, 0}:
                    # LOGGER.info(self.progress_string())
                    pbar = TQDM(enumerate(self.train_loader), total=nb)
                self.tloss = None

                # Send number_batch to RabbitMQ
                if epoch == 0:
                    success = self.send_number_batch_client_id(nb, self.client_id, self.cut_layer, self.tensor_send_ids)
                    if not success:
                        print(f"Không thể gửi number_batch tới queue.")
                if self.model.is_training == True:
                    self.start_thread()
                #Training loop   
                for i, batch in pbar:
                    start_batch_forward_time = time.time()
                    self.run_callbacks("on_train_batch_start")
                    # Warmup
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                    # Forward
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if self.layer_id == 1:
                            data_id = uuid.uuid4()
                            success = self.send_label(data_id, batch)
                            if not success:
                                print(f"Không thể gửi batch {i} tới label_queue.")

                        # Forward in task
                        preds = self.model(batch["img"])

                        duration = round(self.model.end_batch_forward_time - start_batch_forward_time, 2)
                        Utils.log_to_csv('./log/log_time.csv', {
                            'layer_id': self.layer_id,
                            'client_id': self.client_id,
                            'epoch': epoch+1,
                            'forward/backward/end_epoch': 'forward',
                            'duration': round(duration, 2)
                        })

                    # Log
                    if RANK in {-1, 0}:
                        pbar.set_description(f"{epoch + 1}/{self.epochs}")
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)
                            
                    self.run_callbacks("on_train_batch_end")
                self.wait_all_backward(expected_num=nb)

                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

                end_epoch_time = time.time()
                duration = round(end_epoch_time - start_epoch_time, 2)
                Utils.log_to_csv('./log/log_time.csv', {
                    'layer_id': self.layer_id,
                    'client_id': self.client_id,
                    'epoch': epoch+1,
                    'forward/backward/end_epoch': 'end_epoch',
                    'duration': round(duration, 2)
                })
                self.run_callbacks("on_train_epoch_end")
                if RANK in {-1, 0}:
                    final_epoch = epoch + 1 >= self.epochs
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                    # Stopper
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save model
                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                # Scheduler
                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.run_callbacks("on_fit_epoch_end")
                self._clear_memory(0.5)  # clear if memory utilization > 50%

                # Early Stopping
                if RANK != -1:  # if DDP training
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                    self.stop = broadcast_list[0]
                if self.stop:
                    break  # must break all DDP ranks
                epoch += 1
        else:
            LOGGER.info(f"START TRAINING IN CLIENT 2")
            while True:
                self.epoch = epoch
                self.run_callbacks("on_train_epoch_start")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                    self.scheduler.step()

                self._model_train()
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)

                fake_batches = [None] * nb
                pbar = enumerate(fake_batches)

                if RANK in {-1, 0}:
                    LOGGER.info(self.progress_string())
                    pbar = TQDM(enumerate(fake_batches), total=nb)
                self.tloss = None

                #Training loop
                for i, batch in pbar:
                    start_batch_forward_time = time.time()
                    self.run_callbacks("on_train_batch_start")
                    # Warmup
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                    batch = self.wait_for_batch()
                    # Forward
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= world_size
                        self.tloss = (
                            (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                        )

                    duration = round(self.model.end_batch_forward_time - start_batch_forward_time, 2)
                    Utils.log_to_csv('./log/log_time.csv', {
                        'layer_id': self.layer_id,
                        'client_id': self.client_id,
                        'epoch': epoch+1,
                        'forward/backward/end_epoch': 'forward',
                        'duration': round(duration, 2)
                    })
                    # Backward
                    start_batch_backward_time = time.time()
                    self.scaler.scale(self.loss).backward()

                    if self.layer_id == 2:
                        if hasattr(self.model, 'saved_tensor'):
                            gradient_store = {}
                            for tensor_id, tensor in self.model.saved_tensor.items():
                                if tensor.grad is not None:
                                    print(f"Gradient shape của tensor {tensor_id}: {tensor.grad.shape}")
                                    gradient_store[tensor_id] = tensor.grad
                                else:
                                    print(f"Gradient của tensor {tensor_id} là None")
                            
                            # Send gradients to gradient_queue
                            if gradient_store:
                                data_id = self.model.input_data_id
                                success = self.send_gradient(data_id, gradient_store)
                                if not success:
                                    print(f"Không thể gửi Gradients {i} tới Gradient_queue.")
                        
                        if hasattr(self.model, 'saved_data_store'):
                            for tensor_id, tensor in self.model.saved_data_store.items():
                                if tensor.grad is not None:
                                    print(f"Gradient shape của tensor {tensor_id} (data_store): {tensor.grad.shape}")
                                else:
                                    print(f"Gradient của tensor {tensor_id} (data_store) là None")
                    
                    # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni

                        # Timed stopping
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break
                    
                    # Log time
                    end_batch_backward_time = time.time()
                    duration = round(end_batch_backward_time - start_batch_backward_time, 2)
                    Utils.log_to_csv('./log/log_time.csv', {
                        'layer_id': self.layer_id,
                        'client_id': self.client_id,
                        'epoch': epoch+1,
                        'forward/backward/end_epoch': 'backward',
                        'duration': round(duration, 2)
                    })

                    # Log
                    if RANK in {-1, 0}:
                        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                                *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                                batch["cls"].shape[0],  # batch size, i.e. 8
                                batch["img"].shape[-1],  # imgsz, i.e 640
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)

                    self.run_callbacks("on_train_batch_end")

                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
                self.run_callbacks("on_train_epoch_end")
                if RANK in {-1, 0}:
                    final_epoch = epoch + 1 >= self.epochs
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                    # Validation
                    if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                        self._clear_memory(threshold=0.5)  # prevent VRAM spike
                        self.metrics, self.fitness = self.validate()
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save model
                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                # Scheduler
                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.run_callbacks("on_fit_epoch_end")
                self._clear_memory(0.5)  # clear if memory utilization > 50%

                # Early Stopping
                if RANK != -1:  # if DDP training
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                    self.stop = broadcast_list[0]
                if self.stop:
                    break  # must break all DDP ranks
                epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            # self.final_eval()
            if self.args.plots and self.layer_id != 1:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def start_thread(self):
        """START THREADING"""
        thread = threading.Thread(target=self.check_gradient, daemon=True)
        thread.start()

    def stop_thread(self):
        """STOP THREADING"""
        self.model.is_training = False
        print(f"Thread đã dừng.")

    def send_label(self, data_id, labels):
        queue_name = f'label_queue'
        self.channel.queue_declare(queue_name, durable=False)
        CLIENT_LABEL_MAP = self.build_client_map(self.model.names)
        local_labels = labels["cls"]
        global_labels = [CLIENT_LABEL_MAP[int(l)] for l in local_labels]
        print(f"Local label: {local_labels} -> Global label: {global_labels}")
        labels['cls'] = torch.tensor(global_labels)
        message = pickle.dumps(
            {"data_id": data_id,
            "label": labels}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )
        print(f"Batch {data_id} đã được gửi tới {queue_name}, Kích thước: {len(message)} bytes")
        return True
    
    def send_number_batch_client_id(self, nb = None, client_id = None, client_cut_layer = None, tensor_send_ids = None):
        queue_name = f'number_batch_queue'
        self.channel.queue_declare(queue_name, durable=False)

        message = pickle.dumps(
            {"nb": nb,
             "client_id": client_id,
             "client_cut_layer": client_cut_layer,
             "tensor_send_ids": tensor_send_ids}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )
        print(f"Number batch đã được gửi tới {queue_name}")
        return True
    
    def wait_for_number_batch_client_id(self):
        expected_messages = self.num_client[0]
        total_nb = 0
        received = 0
        while received < expected_messages:
            queue_name = f'number_batch_queue'
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                print("Received data:", received_data)
                nb = received_data["nb"]
                client_id = received_data["client_id"]
                client_cut_layer = received_data["client_cut_layer"]
                tensor_send_ids = received_data["tensor_send_ids"]
                if nb is not None and client_id is not None:
                    total_nb += nb
                    self.client_ids.append(client_id) 
                    self.cut_layer_ids.append(client_cut_layer) 
                    self.tensor_send_ids.append(tensor_send_ids) 
                    received += 1
            else:
                time.sleep(0.5)
        self.channel.queue_delete(queue=queue_name)
        return total_nb
    
    def wait_for_batch(self):
        while True:
            queue_name = f'label_queue'
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                data_id = received_data["data_id"]
                data = received_data["label"]
                print(f"Received BATCH with data_id: {data_id}")
                return data
            else:
                # print("No data received yet, waiting...")
                time.sleep(1)

    def send_gradient(self, data_id, gradients):
        # queue_name = f'gradient_queue_{self.layer_id - 1}'
        client_id = data_id.split("_")[0]
        queue_name = f'gradient_queue_{client_id}'

        self.channel.queue_declare(queue_name, durable=False)

        message = pickle.dumps(
            {"data_id": data_id,
            "gadients": gradients}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )

        print(f"Gradients {data_id} đã được gửi tới {queue_name}, Kích thước: {len(message)} bytes")
        return True

    def wait_gradient(self):
        """
        Wait for gradient data from the gradient_queue.

        Returns:
            tuple: (success_flag, grad4, grad6, grad10)
        """
        while True:
            queue_name = f'gradient_queue_{self.layer_id}'
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame and body:
                try:
                    received_data = pickle.loads(body)
                    data_id = received_data.get('data_id')
                    gradient_store = received_data.get('gadients', {})

                    if not isinstance(gradient_store, dict):
                        raise ValueError("Received 'gadients' is not a valid dictionary")

                    gradient_dict = {}

                    for tensor_id in self.tensor_send_ids:
                        grad = gradient_store.get(tensor_id)
                        if grad is None:
                            raise ValueError(f"Missing gradient for tensor_id {tensor_id}")
                        if not isinstance(grad, torch.Tensor):
                            raise ValueError(f"Gradient for tensor_id {tensor_id} is not a valid tensor")
                        print(f"Received gradient for tensor_id {tensor_id}, shape: {grad.shape}")
                        gradient_dict[tensor_id] = grad

                    return True, gradient_dict

                except (pickle.UnpicklingError, ValueError) as e:
                    print(f"Error processing gradient queue data: {e}")
                    time.sleep(0.5)
            else:
                # print("No gradient data received yet, waiting...")
                time.sleep(0.5)

    def check_gradient(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(host=self.address, credentials=credentials)
        thread_connection = pika.BlockingConnection(parameters)
        thread_channel = thread_connection.channel()
        queue_name = f'gradient_queue_{self.client_id}'
        while self.model.is_training:
            try:
                if thread_channel is not None and thread_channel.is_open:
                    method_frame, header_frame, body = thread_channel.basic_get(queue=queue_name, auto_ack=True)
                    if method_frame and body:
                        start_batch_backward_time = time.time()
                        received_data = pickle.loads(body)
                        data_id = received_data.get('data_id')
                        print("\nDATA_ID backward: ", data_id)
                        gradient_store = received_data.get('gadients', {})
                        if not isinstance(gradient_store, dict):
                            raise ValueError("Received 'gadients' is not a valid dictionary")
                        
                        gradient_dict = {}

                        for tensor_id in self.tensor_send_ids:
                            grad = gradient_store.get(tensor_id)
                            if grad is None:
                                raise ValueError(f"Missing gradient for tensor_id {tensor_id}")
                            if not isinstance(grad, torch.Tensor):
                                raise ValueError(f"Gradient for tensor_id {tensor_id} is not a valid tensor")
                            print(f"Received gradient for tensor_id {tensor_id}, shape: {grad.shape}")
                            gradient_dict[tensor_id] = grad

                        # Backward
                        tensor_list = [self.model.data_store[t_id] for t_id in gradient_dict.keys()]
                        grad_list = [gradient_dict[t_id] for t_id in gradient_dict.keys()]
                        torch.autograd.backward(tensor_list, grad_list)
                        self.count_batch += 1

                        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                        # self.optimizer_step()
                        for g in self.optimizer.param_groups:
                            for p in g['params']:
                                if p.grad is not None:
                                    p.grad.data = p.grad.data.float()  # đảm bảo FP32
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break

                        # Log
                        end_batch_backward_time = time.time()
                        duration = round(end_batch_backward_time - start_batch_backward_time, 2)
                        Utils.log_to_csv('./log/log_time.csv', {
                            'layer_id': self.layer_id,
                            'client_id': self.client_id,
                            'epoch': self.epoch + 1,
                            'forward/backward/end_epoch': 'backward',
                            'duration': round(duration, 2)
                        })
                else:
                    print("Thread channel is None or closed")
            except Exception as e:
                print("Error in check_gradient thread:", e)
                break
            time.sleep(1)

        thread_channel.close()
        thread_connection.close()

    def wait_all_backward(self, expected_num):
        with self.condition:
            while self.count_batch < expected_num:
                # print(f"Waiting... Current: {self.count_batch}/{expected_num}")
                self.condition.wait(timeout=0.5)

            print("Enough gradients received. Continue training.")
            self.count_batch = 0


    def get_tensor_send_id (self, cut_layer):
        tensor_send_id = []
        mf_values = []
        layer_indices = []
        for idx, m in enumerate(self.model.model):
            f = m.f
            if f != -1:
                if isinstance(f, int):
                    f = [f]
                for fi in f:
                    if fi != -1:
                        layer_indices.append(idx)
                        mf_values.append(fi)
        mf_values_sorted = sorted(mf_values)

        for value in mf_values_sorted:
            if value < cut_layer:
                tensor_send_id.append(value)

        indices_to_mf = dict(zip(layer_indices, mf_values))
        for idx, val in indices_to_mf.items():
            if idx <=cut_layer:
                tensor_send_id.remove(val)

        tensor_send_id.append(cut_layer)
        print ("SEND tensor id: ", tensor_send_id)
        return tensor_send_id

    def send_epoch_intermediate(self, epoch_intermediate_path = None):
        queue_name = f'Server_queue'
        self.channel.queue_declare(queue_name, durable=False)
        epoch_intermediate_path = str(epoch_intermediate_path).replace("F:\\Do_an\\split_learning", "/app").replace("\\", "/")
        message = pickle.dumps(
            {"action": "VAL_INTER",
             "client_id": self.client_id,
             "layer_id": self.layer_id,
             "epoch": self.epoch,
             "epoch_intermediate": epoch_intermediate_path}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )
        print(f"Epoch intermediate path đã được gửi tới {queue_name}")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        if self.layer_id == 1:
            torch.save(
                {
                    "epoch": self.epoch,
                    "best_fitness": self.best_fitness,
                    "model": None,  # resume and final checkpoints derive from EMA
                    "ema": deepcopy(self.ema.ema).half(),
                    "updates": self.ema.updates,
                    "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                    "train_args": vars(self.args),  # save as dict
                    "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                    # "train_results": self.read_results_csv(),
                    "date": datetime.now().isoformat(),
                    "version": __version__,
                    "license": "AGPL-3.0 (https://ultralytics.com/license)",
                    "docs": "https://docs.ultralytics.com",
                },
                buffer,
            )
        elif self.layer_id == 2:
            torch.save(
                {
                    "epoch": self.epoch,
                    "best_fitness": self.best_fitness,
                    "model": None,  # resume and final checkpoints derive from EMA
                    "ema": deepcopy(self.ema.ema).half(),
                    "updates": self.ema.updates,
                    "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                    "train_args": vars(self.args),  # save as dict
                    "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                    "train_results": self.read_results_csv(),
                    "date": datetime.now().isoformat(),
                    "version": __version__,
                    "license": "AGPL-3.0 (https://ultralytics.com/license)",
                    "docs": "https://docs.ultralytics.com",
                },
                buffer,
            )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0) and self.validate_intermediate == True:
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
            model_path = self.wdir / f"epoch{self.epoch}.pt"
            self.send_epoch_intermediate(model_path)
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint

    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data, self.layer_id)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data
    
    def final_eval(self):
        """Perform final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    if self.layer_id != 1:
                        LOGGER.info(f"\nValidating {f}...")
                        self.validator.args.plots = self.args.plots
                        self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")
    
    def build_client_map(self, local_names):
        GLOBAL_LABELS = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9"
        }
        reverse_global = {v: k for k, v in GLOBAL_LABELS.items()}  # {name: global_id}
        mapping = {}
        for local_id, name in local_names.items():
            mapping[local_id] = reverse_global[name]
        return mapping

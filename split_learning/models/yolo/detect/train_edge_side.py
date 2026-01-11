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
from typing import Dict, Optional, Union, List, Any
from torch import distributed as dist
from torch import nn
from typing import Optional
from ultralytics.utils import LOCAL_RANK, RANK, TQDM, colorstr
from ultralytics import __version__
from ultralytics.data import build_dataloader
from ultralytics.utils.checks import check_amp, check_imgsz
from ultralytics.utils.plotting import plot_results
from src import Utils
from split_learning.communication.Communication import RabbitMQConnection
from split_learning.communication.send_service import SendService
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from split_learning.nn.model_edge_side import Split_Learning_DetectionModel
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import TORCH_2_4, EarlyStopping, ModelEMA, autocast, de_parallel, torch_distributed_zero_first, unset_deterministic
import random
from ultralytics.utils import (
    LOGGER,
    RANK,
    callbacks,
)
from ultralytics.utils.torch_utils import (
    torch_distributed_zero_first,
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
        self.status_train = False
        self.count_batch = 0

        if isinstance(address, RabbitMQConnection):
            self.rabbitmq = address
            self.address = self.rabbitmq.address
            self.username = self.rabbitmq.username
            self.password = self.rabbitmq.password
        else:
            self.rabbitmq = RabbitMQConnection(self.address, self.username, self.password)
            self.rabbitmq.connect()

        self.channel = self.rabbitmq.get_channel()
        self.send_service = SendService(self.rabbitmq)

        if self.layer_id == 1:
            self.condition = threading.Condition()
            self.rabbitmq_thread = RabbitMQConnection(self.address, self.username, self.password)
            self.rabbitmq_thread.connect()
            self.channel_thread = self.rabbitmq_thread.get_channel()
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
        return None
    
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
        self.tensor_send_ids = self.get_tensor_send_id(self.cut_layer)

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
        self.model.send_service = self.send_service
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
                f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
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

                    data_id = uuid.uuid4()
                    self.model.batch_id = data_id
                    self.model.label = label_data

                    # Forward in task
                    preds = self.model(batch["img"])
                self.num_forward += 1
                
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
                        pbar.update(1)
                        pbar.set_postfix({
                            "FORWARD": self.num_forward,
                            "BACKWARD": self.count_batch
                        })
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

    def send_number_batch_client_id(self, nb = None, client_id = None, client_cut_layer = None, tensor_send_ids = None):
        return self.send_service.send_number_batch_client_id(nb, client_id, client_cut_layer, tensor_send_ids)

    def get_tensor_send_id (self, cut_layer):
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
    
    def start_thread(self):
        """START THREADING"""
        thread = threading.Thread(target=self.check_gradient_queue, daemon=True)
        thread.start()

    def stop_thread(self):
        """STOP THREADING"""
        self.model.is_training = False
        print(f"Thread was stopped.")

    def check_gradient_queue(self):
        queue_name = f'gradient_queue_{self.client_id}'
        try:
            while True:
                q = self.rabbitmq_thread.declare_queue(queue_name, durable=False)
                message_count = q.method.message_count

                if message_count > 0:
                    self.backward_flag = True
                else:
                    self.backward_flag = False
                time.sleep(0.5)
        except pika.exceptions.ChannelClosedByBroker:
            print(f"Queue '{queue_name}' không tồn tại trên RabbitMQ server.")
        except Exception as e:
            print(f"Lỗi khi kiểm tra queue: {e}")

    def wait_gradient(self):
        """
        Wait for gradient data from the gradient_queue.

        Returns:
            tuple: (success_flag, grad4, grad6, grad10)
        """
        tensor_send_ids = self.get_tensor_send_id(self.cut_layer)
        while True:
            queue_name = f'gradient_queue_{self.client_id}'
            method_frame, header_frame, body = self.rabbitmq.get_message(queue_name=queue_name, auto_ack=True)
            if method_frame and body:
                try:
                    received_data = pickle.loads(body)
                    data_id = received_data.get('data_id')
                    gradient_store = received_data.get('gradients', {})

                    if not isinstance(gradient_store, dict):
                        raise ValueError("Received 'gradients' is not a valid dictionary")

                    gradient_dict = {}

                    for tensor_id in tensor_send_ids:
                        grad = gradient_store.get(tensor_id)
                        if grad is None:
                            raise ValueError(f"Missing gradient for tensor_id {tensor_id}")
                        if not isinstance(grad, torch.Tensor):
                            raise ValueError(f"Gradient for tensor_id {tensor_id} is not a valid tensor")
                        gradient_dict[tensor_id] = grad

                    return True, gradient_dict

                except (pickle.UnpicklingError, ValueError) as e:
                    print(f"Error processing gradient queue data: {e}")
                    time.sleep(0.5)
            else:
                time.sleep(0.5)
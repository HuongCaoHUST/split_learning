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
from engine.rtdetr.model import Split_Learning_RTDETRDetectionModel
from engine.yolo.train import Split_Learning_DetectionTrainer
from .val import RTDETRDataset, RTDETRValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics import __version__
from ultralytics.utils.checks import check_amp, check_imgsz
from ultralytics.data.utils import check_cls_dataset
from ultralytics.data import build_dataloader
from ultralytics.utils.plotting import plot_results
from engine.yolo.data import check_det_dataset
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

class Split_Learning_RTDETRTrainer(Split_Learning_DetectionTrainer):
    def __init__(self, overrides, client_id=None, layer_id=None, num_client=None, cut_layer=None, address=None, username=None, password=None, load_partial_model=False, FedAvg=False):
     
        super().__init__(overrides=overrides, client_id=client_id, layer_id=layer_id, num_client=num_client, cut_layer=cut_layer,
            address=address, username=username, password=password, load_partial_model=load_partial_model, FedAvg=FedAvg)
    
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
    
    def get_model(self, cfg: Optional[dict] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = Split_Learning_RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1,
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
    def build_dataset(self, img_path: str, mode: str = "val", batch: Optional[int] = None):
        """
        Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """Return a DetectionValidator suitable for RT-DETR model validation."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
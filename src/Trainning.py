import time
import uuid
import pickle
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from ultralytics.models.yolo import YOLO
import src.Log


class Trainning:
    def __init__(self, client_id, layer_id, channel, device, event_time=False):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.data_count = 0

        self.event_time = event_time
        self.time_event = []

    def train_on_first_layer(self, model_path, dataset_path):
        
        print(f"Model path is: {model_path}")
        print(f"Dataset path is: {dataset_path}")

        while True:
            model = YOLO(model_path)
            model.train(
                data=dataset_path,
                epochs=3,
                imgsz=640,
                batch=16
            )

        # Finish epoch training, send notify to server
        src.Log.print_with_color("[>>>] Finish training!", "red")

    def train_on_device(self, model_path, dataset_path):
        self.data_count = 0
        if self.layer_id == 1:
            result = self.train_on_first_layer(model_path, dataset_path)
        # elif self.layer_id == num_layers:
        #     result = self.train_on_last_layer()

        if self.event_time:
            src.Log.print_with_color(f"Training time events {self.time_event}", "yellow")
        return result, self.data_count

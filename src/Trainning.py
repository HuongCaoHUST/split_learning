import time
import uuid
import pickle
import pika
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import threading
from ultralytics.models.yolo.detect import DetectionTrainer
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
        self.best_model = None
    

    def send_to_server(self, message):
        print (f"[>>>] Client {self.client_id} send message to server: {message}")
        self.channel.queue_declare('Server_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='Server_queue',
                                   body=pickle.dumps(message))

    def train_on_first_layer(self, model_path, dataset_path, num_client, cut_layer, epochs, batch_size, address = None, username = None, password = None):
        src.Log.print_with_color("--- START TRAINING FIRST LAYER ---", "green")
        args = dict(model=model_path,
                    data=dataset_path,
                    num_client=num_client,
                    epochs=epochs,
                    batch=batch_size,
                    save_period = 1,
                    client_id=self.client_id,
                    layer_id=self.layer_id,
                    cut_layer=cut_layer,
                    address=address,
                    username=username,
                    password=password,
                    channel=self.channel)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        self.best_model = trainer.best
        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                           "message": "Finish training!", "validate": None}
        # Finish epoch training, send notify to server
        self.send_to_server(notify_data)
        src.Log.print_with_color("[>>>] Finish training!", "red")

        broadcast_queue_name = f'reply_{self.client_id}'
        while True:  # Wait for broadcast
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    return True
                break
            time.sleep(0.5)

    def train_on_last_layer(self, model_path, dataset_path, num_client, cut_layer, epochs, batch_size, address = None, username = None, password = None):
        queue_name = f'label_queue'
        result = True
        self.channel.queue_declare(queue=queue_name, durable=False)
        self.channel.queue_declare(queue="number_batch_queue", durable=False)
        self.channel.basic_qos(prefetch_count=10)
        print('Waiting for intermediate output. To exit press CTRL+C')

        src.Log.print_with_color("--- START TRAINING SECOND LAYER ---", "green")
        args = dict(model=model_path,
                    data=dataset_path,
                    num_client=num_client,
                    epochs=epochs,
                    batch=batch_size,
                    save_period = 1,
                    # optimizer='AdamW',
                    client_id=self.client_id,
                    close_mosaic = 5,
                    layer_id=self.layer_id,
                    cut_layer=cut_layer,
                    address=address,
                    username=username,
                    password=password,
                    channel=self.channel)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        self.best_model = trainer.best
        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                           "message": "Finish training!", "validate": None}
        # Finish epoch training, send notify to server
        self.send_to_server(notify_data)
        src.Log.print_with_color("[>>>] Finish training!", "red")

        # Check training process
        broadcast_queue_name = f'reply_{self.client_id}'
        while True:  # Wait for broadcast
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    return True
                break
            time.sleep(0.5)
                    
    def train_on_device(self, model_path, dataset_path, num_client,cut_layer, epochs, batch_size, address, username, password):
        self.data_count = 0
        if self.layer_id == 1:

            # Create gradient queue
            forward_queue_name = f'gradient_queue_{self.client_id}'
            self.channel.queue_declare(queue=forward_queue_name, durable=False)
            self.channel.basic_qos(prefetch_count=10)

            result = self.train_on_first_layer(model_path, dataset_path, num_client,cut_layer, epochs, batch_size, address, username, password)

        elif self.layer_id == 2:
            # Create intermediate queue
            forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'
            self.channel.queue_declare(queue=forward_queue_name, durable=False)
            self.channel.basic_qos(prefetch_count=10)

            # Create label queue
            forward_queue_name = f'label_queue'
            self.channel.queue_declare(queue=forward_queue_name, durable=False)
            self.channel.basic_qos(prefetch_count=10)
            
            result = self.train_on_last_layer(model_path, dataset_path, num_client,cut_layer, epochs, batch_size, address, username, password)

        if self.event_time:
            src.Log.print_with_color(f"Training time events {self.time_event}", "yellow")
        return result, self.best_model

import os
import time
import pika
import pickle
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import requests
from ultralytics.models.yolo.detect import DetectionValidator
from requests.auth import HTTPBasicAuth
from ultralytics import YOLO
import src.Model
import src.Log
import src.Utils
import src.Validation

num_labels = 10


def delete_old_queues(address, username, password):
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue") or queue_name.startswith("rpc_queue") or queue_name.startswith("label_queue"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to delete queue '{queue_name}': {e}", "yellow")
            else:
                try:
                    http_channel.queue_purge(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to purge queue '{queue_name}': {e}", "yellow")

        connection.close()
        return True
    else:
        src.Log.print_with_color(
            f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}", "yellow")
        return False

class Server:
    def __init__(self, config_dir):
        with open(config_dir, 'r') as file:
            config = yaml.safe_load(file)

        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        delete_old_queues(address, username, password)

        # Clients
        self.total_clients = config["server"]["clients"]
        self.batch_size = config["learning"]["batch-size"]
        self.lr = config["learning"]["learning-rate"]
        self.momentum = config["learning"]["momentum"]
        self.control_count = config["learning"]["control-count"]
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        # Model
        self.model_path = config["model"]["model_path"]
        self.cut_layer = config["model"]["cut_layer"]

        #Dataset
        self.dataset_path = config["dataset"]["dataset_path"]

        log_path = config["log_path"]

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue='Server_queue')
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='Server_queue', on_message_callback=self.on_request)
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info("Application start")

        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

    def start(self):
        self.channel.start_consuming()

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]

        if (str(client_id), layer_id) not in self.list_clients:
            self.list_clients.append((str(client_id), layer_id))

        if action == "REGISTER":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            self.register_clients[layer_id - 1] += 1

            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_to_clients()

        elif action == "NOTIFY":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # for (client_id, layer_id) in self.list_clients:
            message = {"action": "PAUSE",
                        "message": "Pause training and please send your parameters",
                        "parameters": None}
            # self.send_to_client(client_id, pickle.dumps(message))
            src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
            response = {"action": "STOP",
                        "message": "Stop training!",
                        "parameters": None}
            self.send_to_client(client_id, pickle.dumps(response))

        # self.notify_to_clients(start=False)
        # sys.exit()             
        elif action == "UPDATE":
            best = message["best"]
            client_id = message["client_id"]
            src.Log.print_with_color(f"[<<<] Received best model from client: {best}", "blue")
            if layer_id == 2:
                print("BEST.pt:", best)
                args = dict(model=best, data="F:/Do_an/split_learning/datasets/coco128.yaml")
                validator = DetectionValidator(args=args)
                validator()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_to_clients(self, start=True, register=True):

        src.Log.print_with_color(f"notify_client", "red")
        for (client_id, layer_id) in self.list_clients:
            if start:
                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "num_layers": len(self.total_clients),
                            "model_path": self.model_path,
                            "dataset_path": self.dataset_path,
                            "cut_layer": self.cut_layer,
                            "control_count": self.control_count,
                            "batch_size": self.batch_size,
                            "lr": self.lr,
                            "momentum": self.momentum}
            # else:
            #     src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
            #     response = {"action": "STOP",
            #                 "message": "Stop training!",
            #                 "parameters": None}
            self.time_start = time.time_ns()
            src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
            self.send_to_client(client_id, pickle.dumps(response))

    def send_to_client(self, client_id, message):
        reply_channel = self.connection.channel()
        reply_queue_name = f'reply_{client_id}'
        reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def merge_yolo(client1_pt, client2_pt, yaml_file, cut_layer, save_file='merged.pt'):
        # Load model cấu hình gốc
        model = YOLO("yolo11n.yaml")

        # Load checkpoints
        ckpt1 = torch.load(client1_pt, map_location='cpu')
        ckpt2 = torch.load(client2_pt, map_location='cpu')
        state1 = ckpt1['model'] if 'model' in ckpt1 else ckpt1
        state2 = ckpt2['model'] if 'model' in ckpt2 else ckpt2

        # State dict gốc
        full_state = model.model.state_dict()

        # Merge
        for k in state1:
            if k.startswith('model.') and int(k.split('.')[1]) <= cut_layer:
                full_state[k] = state1[k]

        for k in state2:
            if k.startswith('model.') and int(k.split('.')[1]) > cut_layer:
                full_state[k] = state2[k]

        # Load lại
        model.model.load_state_dict(full_state, strict=False)

        # Save file pt đầy đủ
        torch.save({'model': model.model.state_dict()}, save_file)
        print(f"✅ Đã gộp xong. File lưu tại: {save_file}")
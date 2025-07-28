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

def delete_old_queues(address, username, password):
    url = f'http://{address}:15672/api/queues'

    while True:
        try:
            response = requests.get(url, auth=HTTPBasicAuth(username, password))
            if response.status_code == 200:
                break
            else:
                src.Log.print_with_color(f"⚠️ Waiting for RabbitMQ API... Status: {response.status_code}", "yellow")
        except requests.exceptions.ConnectionError:
            src.Log.print_with_color("⏳ Waiting for RabbitMQ HTTP API to be ready...", "yellow")
        time.sleep(1)

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

        self.address = config["rabbit"]["address"]
        self.username = config["rabbit"]["username"]
        self.password = config["rabbit"]["password"]
        delete_old_queues(self.address, self.username, self.password)

        # Clients
        self.total_clients = config["server"]["clients"]
        self.batch_size = config["learning"]["batch-size"]
        self.lr = config["learning"]["learning-rate"]
        self.momentum = config["learning"]["momentum"]
        self.control_count = config["learning"]["control-count"]
        self.epochs = config["learning"]["epochs"]

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        # Model
        self.model_path = config["model"]["model_path"]
        self.cut_layer = config["model"]["cut_layer"]
        self.output_model = config["model"]["output_model"]
        self.best_model_layer_1 = []
        self.best_model_2 = None

        #Dataset
        self.dataset_path = config["dataset"]["dataset_path"]

        log_path = config["log_path"]

        self.connect()

        self.channel.queue_declare(queue='Server_queue')
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='Server_queue', on_message_callback=self.on_request)
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info("Start Training")

        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        while True:
            try:
                self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
                self.channel = self.connection.channel()
                break
            except pika.exceptions.AMQPConnectionError:
                print("⏳ Waiting for RabbitMQ to be ready...")
                time.sleep(1)
    
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
            docker = message["docker"]
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
            if layer_id == 1:
                self.best_model_layer_1.append(best)
                print("BEST_layer_1.pt:", best)
            if layer_id == 2:
                self.best_model_2 = best
                print("BEST_2.pt:", self.best_model_2)
                merge_model = self.merge_yolo_models()
                args = dict(model=merge_model, data="/app/datasets/livingroom_2_1_for_docker.yaml")
                validator = DetectionValidator(args=args)
                validator()
                sys.exit()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_to_clients(self, start=True, register=True):

        src.Log.print_with_color(f"notify_client", "red")
        print("self.list_client: ", self.list_clients)
        layer1_clients = [(client_id, layer_id) for client_id, layer_id in self.list_clients if layer_id == 1]

        print("layer1_client: ", layer1_clients)

        dataset_index = 0
        for (client_id, layer_id) in self.list_clients:
            
            if layer_id == 1:
                dataset_path = self.dataset_path[dataset_index]
                dataset_index += 1
            else:
                dataset_path = self.dataset_path[0]
            
            if layer_id == 2:
                dataset_path = "/app/datasets/livingroom_concat_docker.yaml"

            if start:
                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "num_client": self.total_clients,
                            "model_path": self.model_path,
                            "dataset_path": dataset_path,
                            "cut_layer": self.cut_layer,
                            "control_count": self.control_count,
                            "epochs": self.epochs,
                            "batch_size": self.batch_size,
                            "lr": self.lr,
                            "momentum": self.momentum}

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
    
    def merge_yolo_models(self):
        if self.total_clients[0] == 1:
            model1 = YOLO(self.best_model_layer_1[0])
            model2 = YOLO(self.best_model_2)
            output_path = self.output_model

            state_dict1 = model1.model.state_dict()
            state_dict2 = model2.model.state_dict()

            new_state_dict = state_dict2.copy()

            for k in state_dict1.keys():
                if k.startswith("model."):
                    try:
                        layer_num = int(k.split('.')[1])
                        if layer_num <= self.cut_layer:
                            new_state_dict[k] = state_dict1[k]
                    except:
                        pass

            model2.model.load_state_dict(new_state_dict)

            model2.save(output_path)

            print(f"Đã ghép xong model và lưu tại: {output_path}")
            return output_path
        else:
            state_dicts = []
            output_path = self.output_model
            for model_path in self.best_model_layer_1:
                model = YOLO(model_path)
                state_dicts.append(model.model.state_dict())
            
            # Average weights
            avg_state_dict = {}
            for key in state_dicts[0].keys():
                if key.startswith("model."):
                    try:
                        layer_num = int(key.split('.')[1])
                        if layer_num <= self.cut_layer:
                            weights = [sd[key] for sd in state_dicts]
                            avg_weight = sum(weights) / len(weights)
                            avg_state_dict[key] = avg_weight
                    except:
                        pass

            model2 = YOLO(self.best_model_2)
            state_dict2 = model2.model.state_dict()
            new_state_dict = state_dict2.copy()
            new_state_dict.update(avg_state_dict)

            model2.model.load_state_dict(new_state_dict)
            model2.save(output_path)

            print(f"Đã ghép xong model và lưu tại: {output_path}")
            return output_path
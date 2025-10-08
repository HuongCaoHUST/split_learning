import os
import time
import pika
import pickle
import sys
import yaml
import numpy as np
import requests
import random
from ultralytics.data.utils import check_det_dataset
from requests.auth import HTTPBasicAuth
import src.Log
import src.Utils
from src.Validation import ModelValidator
from src.Utils import split_dataset

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
                    "gradient_queue") or queue_name.startswith("label_queue"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to delete queue '{queue_name}': {e}", "yellow")
            # else:
            #     try:
            #         http_channel.queue_purge(queue=queue_name)
            #         src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
            #     except Exception as e:
            #         src.Log.print_with_color(f"Failed to purge queue '{queue_name}': {e}", "yellow")

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
        self.epochs = config["learning"]["epochs"]
        self.worker = config["learning"]["worker"]

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        # Model
        self.task = config["model"]["task"]
        self.model_path = config["model"]["model_path"]
        # self.model_path = self.get_model_path(self.task)
        self.cut_layer = config["model"]["cut_layer"]
        if len(self.total_clients) > len(self.cut_layer):
            self.cut_layer = [self.cut_layer[0] for _ in range(len(self.total_clients))]
        self.valid_epoch_model = 1 if config["model"]["valid_epoch_model"] else -1
        self.hybrid_training = config["model"]["hybrid_training"]
        self.output_model = config["model"]["output_model"]

        #Dataset
        self.dataset_path = config["dataset"]["dataset_path"] if not config["dataset"]["iid_datasets"] else self.random_dataset(num_clients=self.total_clients[0])
        # self.dataset_path = src.Utils.split_dataset(yaml_path=self.dataset_path[0], num_client=self.total_clients[0])
        print("Dataset paths for clients:", self.dataset_path)

        if len(self.total_clients) > len(self.dataset_path):
            self.dataset_path = [self.dataset_path[0] for _ in range(len(self.total_clients))]
        # self.nb_client = src.Utils.check_dataset(self.dataset_path, self.batch_size)

        self.val_function = ModelValidator(
            total_client=self.total_clients,
            hybrid_training=self.hybrid_training,
            cut_layer=self.cut_layer,
            best_model_layer_1=[],
            best_model_2=[],
            epoch_model_layer_1=[],
            epoch_model_layer_2=[],
            dataset_path=self.dataset_path,
            output_model=self.output_model
        )

        self.concatenate_datasets = config["dataset"]["concatenate_datasets"]
        if self.concatenate_datasets == True and self.total_clients[0] >1:
            self.concatenate_func()

        log_path = config["log_path"]

        self.connect()

        self.channel.queue_declare(queue='Server_queue')
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='Server_queue', on_message_callback=self.on_request)
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info("Start Training")
        src.Utils.init_csv(f"{log_path}/log/log_validation.csv", headers=["epoch", "precision", "recall", "mAP50", "mAP50-95"])

        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")
        

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        while True:
            try:
                self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
                self.channel = self.connection.channel()
                break
            except pika.exceptions.AMQPConnectionError:
                time.sleep(1)
    
    def start(self):
        self.channel.start_consuming()

    def concatenate_func(self):
        self.nc_list_cumulative = []
        cumulative = 0
        for i, path in enumerate(self.dataset_path):
            if i == self.total_clients[0]:
                break
            result = check_det_dataset(path)
            nc = result['nc']
            cumulative += nc
            self.nc_list_cumulative.append(cumulative)
        print ("CUMULATIVE: ", self.nc_list_cumulative)

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

        elif action == "VAL_INTER":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            client_id = message["client_id"]
            layer_id = message["layer_id"]
            layer_map = {
                1: self.val_function.epoch_model_layer_1,
                2: self.val_function.epoch_model_layer_2
            }
            if layer_id in layer_map:
                layer_map[layer_id].append(message["epoch_intermediate"])
         
        elif action == "UPDATE":
            client_id = message["client_id"]
            virtual_machine=message["vm"]
            if layer_id == 1 and not virtual_machine:
                best = message["best"]
                src.Log.print_with_color(f"[<<<] Received best model from client: {best}", "blue")
                self.val_function.best_model_layer_1.append(best)
                print("BEST_layer_1.pt:", best)
            elif layer_id == 1 and virtual_machine:
                best = message["best"]
                best = src.Utils.save_model_file(best, best_dir="./best_model_vm")
                src.Log.print_with_color(f"[<<<] Received best model from client: {best}", "blue")
                self.val_function.best_model_layer_1.append(best)
            
            elif layer_id == 2:
                best = message["best"]
                src.Log.print_with_color(f"[<<<] Received best model from client: {best}", "blue")
                self.val_function.best_model_2.append(best)
                print("BEST_2.pt:", self.val_function.best_model_2)
                if len(self.val_function.best_model_layer_1) == self.total_clients[0] and len(self.val_function.best_model_2) == self.total_clients[1]:
                    self.val_function.validate_best_model()
                sys.exit()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_to_clients(self, start=True, register=True):

        src.Log.print_with_color(f"notify_client", "red")
        print("self.list_client: ", self.list_clients)
        self.layer1_clients = [(client_id, layer_id) for client_id, layer_id in self.list_clients if layer_id == 1]
        print("layer1_client: ", self.layer1_clients)

        self.layer1_clients_id = [client_id for client_id, layer_id in self.list_clients if layer_id == 1]
        print("layer1_1_client: ", self.layer1_clients_id)

        # self.data_distribution = dict(zip(self.layer1_clients_id, self.nb_client))
        # print("data_distribution: ", self.data_distribution)

        dataset_index = 0
        for (client_id, layer_id) in self.list_clients:
            if start:
                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "num_client": self.total_clients,
                            "task": self.task,
                            "epochs": self.epochs,
                            "batch_size": self.batch_size,
                            "lr": self.lr,
                            "momentum": self.momentum,
                            "worker": self.worker,
                            "valid_epoch_model": self.valid_epoch_model}
                
                if layer_id == 1:
                    response["model_path"] = self.model_path[0]
                    response["cut_layer"] = self.cut_layer[dataset_index]
                    response["dataset_path"] = self.dataset_path[dataset_index]
                    if self.concatenate_datasets and dataset_index !=0:
                        delta_nc = self.nc_list_cumulative[dataset_index - 1]
                        response["concatenate_datasets"] = True
                        response["delta_nc"] = delta_nc
                    dataset_index += 1
                elif layer_id == 2:
                    response["model_path"] = self.model_path[1]
                    response["cut_layer"] = self.cut_layer[0]
                    response["dataset_path"] = self.dataset_path[0]

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

    def random_dataset(self, num_clients):
        dataset_path = "./datasets/mnist_yolo_cls_dirichlet"
        all_clients = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        selected_clients = random.sample(all_clients, num_clients)

        print("✅ Selected clients:", selected_clients)
        selected_paths = [os.path.join(dataset_path, c) for c in selected_clients]
        return selected_paths
    
    def get_model_path(self, task):
        MODEL_PATH = {
            "detect": "./yolo11n.pt",
            "segment": "./yolo11n-seg.pt",
            "classify": "./yolo11n-cls.pt",
        }
        return MODEL_PATH.get(task, "./yolo11n.pt")
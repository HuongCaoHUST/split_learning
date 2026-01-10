import time
import pickle
import pika
import random
import glob
from pathlib import Path

import requests
from torch import nn
import src.Utils
import src.Log
import mlflow

from ultralytics.data.utils import check_det_dataset, IMG_FORMATS


class Client:
    def __init__(self, client_id, layer_id, address, username, password, train_func, device, virtual_machine=False):
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.device = device
        self.train_func = train_func
        self.virtual_machine = virtual_machine
        print(f"Client {self.client_id} initialized with layer {self.layer_id} on device {self.device}")
        self.connect()

        mlflow.set_tracking_uri("http://14.225.254.18:5000")
        mlflow.set_experiment("Split_Learning")
    
    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        self.response = None
        self.channel.queue_declare('Server_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='Server_queue',
                                   body=pickle.dumps(message))
        return self.response
    
    def wait_response(self):
        status = True
        reply_queue_name = f'reply_{self.client_id}'
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                status = self.response_message(body)
                break
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        action = self.response["action"]
        num_round = self.response.get("num_round", 1)
        print("Number of rounds:", num_round)
        model_path = self.response.get("model_path")
        dataset_path = self.response.get("dataset_path")
        cut_layer = self.response.get("cut_layer")
        task = self.response.get("task")
        epochs = self.response.get("epochs")
        batch_size = self.response.get("batch_size")
        num_client = self.response.get("num_client")
        worker = self.response.get("worker")
        load_partial_model = self.response.get("load_partial_model")
        valid_epoch_model = self.response.get("valid_epoch_model")
        run_id = self.response.get("run_id")
        if action == "START":
            src.Log.print_with_color(f"[<<<] Client received: {self.response}", "blue")
            if self.layer_id == 1:
                self.register_node_http(self.client_id, run_id=run_id, num_images=self._count_dataset_labels(dataset_path))
                result, best = self.train_func(model_path, dataset_path, num_client, cut_layer, num_round, task, epochs, batch_size, worker, self.address, self.username, self.password, load_partial_model, valid_epoch_model)
            if self.layer_id == 2:
                self.register_node_http(self.client_id, run_id=run_id)
                with mlflow.start_run(run_id=run_id):
                    result, best = self.train_func(model_path, dataset_path, num_client, cut_layer, num_round, task, epochs, batch_size, worker, self.address, self.username, self.password, load_partial_model, valid_epoch_model)
            
            if self.virtual_machine:
                file_data = src.Utils.read_file(best)
                data = {"action": "UPDATE", "client_id": self.client_id, "layer_id": self.layer_id,
                        "result": result, "message": "Sent parameters to Server", "vm": self.virtual_machine, "best": file_data}
            else:
                best = str(best).replace("F:\\Do_an\\split_learning", "/app").replace("\\", "/")
                data = {"action": "UPDATE", "client_id": self.client_id, "layer_id": self.layer_id,
                        "result": result, "message": "Sent parameters to Server", "vm": self.virtual_machine, "best": best}
            
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "STOP":
            print("Training completed. Client stopping.")
            return False

    def register_node_http(self, client_id, run_id, num_images=None, host="14.225.254.18", port=8000):
        """
        Hàm đăng ký node qua API HTTP.
        """
        url = f"http://{host}:{port}/register"
        payload = {
            "action": "REGISTER",
            "client_id": str(client_id),
            "run_id": str(run_id)
        }
        if num_images is not None:
            payload["number_images"] = num_images
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                src.Log.print_with_color(f"API Register success: {response.json()}", "green")
            else:
                src.Log.print_with_color(f"API Register failed: {response.status_code} {response.text}", "red")
        except Exception as e:
            src.Log.print_with_color(f"API Register error: {e}", "red")

    def _count_dataset_labels(self, dataset_path):
        """Checks a dataset YAML file, derives label paths, and counts label files."""
        if not dataset_path:
            return

        try:
            data = check_det_dataset(dataset_path)  # Load and check dataset

            # Count labels in the training set
            train_path = data.get('train')
            if train_path:
                label_files = []
                # Handle single path string or list of paths
                paths = [train_path] if isinstance(train_path, str) else train_path
                for p in paths:
                    # Derive label path from image path by replacing 'images' with 'labels'
                    label_p = Path(str(p).replace("images", "labels"))
                    if label_p.is_dir():
                        label_files.extend(label_p.rglob("*.txt"))
                train_labels = len(label_files)
                

            # Count labels in the validation set
            val_path = data.get('val')
            if val_path:
                label_files = []
                # Handle single path string or list of paths
                paths = [val_path] if isinstance(val_path, str) else val_path
                for p in paths:
                    # Derive label path from image path by replacing 'images' with 'labels'
                    label_p = Path(str(p).replace("images", "labels"))
                    if label_p.is_dir():
                        label_files.extend(label_p.rglob("*.txt"))
                val_labels = len(label_files)

            return train_labels

        except Exception as e:
            print(f"Lỗi khi xử lý dataset: {e}")
import time
import pickle
import pika
import random
from torch import nn
import src.Utils
import src.Log

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

        if action == "START":
            src.Log.print_with_color(f"[<<<] Client received: {self.response}", "blue")
            for round in range(num_round):
                src.Log.print_with_color(f"--- START ROUND {round + 1} / {num_round} ---", "green")
                if self.layer_id == 1:
                    result, best = self.train_func(model_path, dataset_path, num_client, cut_layer, task, epochs, batch_size, worker, self.address, self.username, self.password, load_partial_model, valid_epoch_model)
                if self.layer_id == 2:
                    result, best = self.train_func(model_path, dataset_path, num_client, cut_layer, task, epochs, batch_size, worker, self.address, self.username, self.password, load_partial_model, valid_epoch_model)
                
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
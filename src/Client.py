import time
import pickle
import pika
import random
import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn

import src.Log
import src.Model

class Client:
    def __init__(self, client_id, layer_id, address, username, password, train_func, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.device = device
        self.train_func = train_func
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
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        action = self.response["action"]
        model_path = self.response.get("model_path")
        dataset_path = self.response.get("dataset_path")
        cut_layer = self.response.get("cut_layer")
        print("CUT_LAYER:", cut_layer)
        src.Log.print_with_color(f"[<<<] Client received: {self.response}", "blue")
        if action == "START":
            if self.layer_id == 1:
                result, size = self.train_func(model_path, dataset_path)
            if self.layer_id == 2:
                result, size = self.train_func(model_path, dataset_path)
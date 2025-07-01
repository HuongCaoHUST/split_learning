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
    def __init__(self, client_id, layer_id, address, username, password, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.device = device
        print(f"Client {self.client_id} initialized with layer {self.layer_id} on device {self.device}")
        self.connect()
    
    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(host=self.address, credentials=credentials)
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        
        self.channel.queue_declare(queue=f"client_{self.client_id}_layer_{self.layer_id}", durable=True)
        print(f"Client {self.client_id} connected to server at {self.address}")
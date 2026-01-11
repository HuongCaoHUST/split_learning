import time
import pickle
import pika
from tqdm import tqdm
from split_learning.models.yolo.detect.train_edge_side import Split_Learning_Edge_DetectionTrainer
from split_learning.models.yolo.detect.train_server_side import  Split_Learning_Server_DetectionTrainer
import src.Log
from ultralytics import YOLO
import torch
from src.Utils import create_yaml_model
from ultralytics import settings
import mlflow

class Training:
    def __init__(self, client_id, layer_id, channel, device, event_time=False):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.data_count = 0

        self.event_time = event_time
        self.time_event = []
        self.best_model = None
        self.current_round = 1
        
        settings.update({"mlflow": False})
    

    def send_to_server(self, message):
        print (f"[>>>] Client {self.client_id} send message to server: {message}")
        self.channel.basic_publish(exchange='',
                                   routing_key='Server_queue',
                                   body=pickle.dumps(message))

    def train_on_first_layer(self, model_path, dataset_path, num_client, cut_layer, num_round, task, epochs, batch_size, worker, address = None, username = None, password = None, load_partial_model=False, valid_epoch_model = -1):
        src.Log.print_with_color(f"--- START TRAINING FIRST LAYER --- CURREN ROUND: {self.current_round} ---", "green")

        yaml_model = create_yaml_model('yolo11n.yaml', 'yolo11n_custom.yaml', cut_layer=cut_layer)
        TRAINER = {
            "detect": Split_Learning_Edge_DetectionTrainer
        }
        TrainerClass = TRAINER.get(task)
        args = dict(model="yolo11n.pt",
                    data=dataset_path,
                    pretrained="./yolo11n.pt",
                    epochs=epochs,
                    batch=batch_size,
                    project = f'./runs/detect/{self.client_id}',
                    workers = worker,
                    save_period = valid_epoch_model)
        trainer = TrainerClass(overrides=args, client_id=self.client_id,
                                         layer_id=self.layer_id, num_client=num_client,
                                         cut_layer=cut_layer, address=address, username=username, password=password, load_partial_model=load_partial_model)
        trainer.train()
        self.best_model = str(trainer.best)
        if not self.best_model.startswith("./"):
            self.best_model = "./" + self.best_model

        self.last_model = str(trainer.last)
        if not self.last_model.startswith("./"):
            self.last_model = "./" + self.last_model

        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                           "message": "Finish round 1!", "round": self.current_round, "best": self.best_model, "last": self.last_model}
        # Finish epoch training, send notify to server
        self.send_to_server(notify_data)
        # src.Log.print_with_color("[>>>] Finish training!", "red")

        broadcast_queue_name = f'reply_{self.client_id}'
        while True:  # Wait for broadcast
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    return True
                elif received_data["action"] == "CONTINUE":
                    self.current_round += 1
                    print("Continue training next round")
                    print("Fed avg model path:", received_data["model_path"])
                    fed_model_path = received_data["model_path"]
                    args = dict(model=fed_model_path,
                                epochs=epochs,
                                data=dataset_path,
                                batch=batch_size,
                                project = f'./runs/detect/{self.client_id}',
                                workers = worker,
                                save_period = valid_epoch_model,
                                lr0=0.001 * (0.92 ** (self.current_round-1)),
                                warmup_epochs=0)
                    trainer = TrainerClass(overrides=args, client_id=self.client_id, layer_id=self.layer_id, num_client=num_client,
                            cut_layer=cut_layer, address=address, username=username, password=password, load_partial_model=load_partial_model, FedAvg=True)
                    trainer.train()
                    self.best_model = str(trainer.best)
                    if not self.best_model.startswith("./"):
                        self.best_model = "./" + self.best_model

                    self.last_model = str(trainer.last)
                    if not self.last_model.startswith("./"):
                        self.last_model = "./" + self.last_model

                    notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                                    "message": "Finish round 2!", "round": self.current_round, "best": self.best_model, "last": self.last_model}
                    self.send_to_server(notify_data)
            time.sleep(0.5)

    def train_on_last_layer(self, model_path, dataset_path, num_client, cut_layer, num_round, task, epochs, batch_size, worker, address = None, username = None, password = None, load_partial_model=False, valid_epoch_model = -1):
        self.channel.queue_declare(queue='label_queue', durable=False)
        self.channel.queue_declare(queue="number_batch_queue", durable=False)
        self.channel.basic_qos(prefetch_count=10)
        print('Waiting for intermediate output. To exit press CTRL+C')

        src.Log.print_with_color("--- START TRAINING SECOND LAYER ---", "green")

        TRAINER = {
            "detect": Split_Learning_Server_DetectionTrainer
        }
        TrainerClass = TRAINER.get(task)
        args = dict(model="./yolo11n.yaml",
                    data=dataset_path,
                    epochs=epochs,
                    batch=batch_size,
                    project = './runs/detect',
                    workers = worker,
                    save_period = valid_epoch_model,
                    optimizer='SGD')
        
        mlflow.log_params({
            "task": task,
            "client_id": self.client_id,
            "layer_id": self.layer_id,
            "num_client": num_client,
            "cut_layer": cut_layer,
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": worker,
            "pretrained": "./yolo11n.pt",
            "dataset": dataset_path
        })

        trainer = TrainerClass(overrides=args, client_id=self.client_id,
                                         layer_id=self.layer_id, num_client=num_client,
                                         cut_layer=cut_layer, address=address, username=username, password=password, load_partial_model=load_partial_model)
        trainer.train()

        # Log to ML FLow
        save_dir = trainer.save_dir
        results_csv = save_dir / "results.csv"
        src.Utils.log_results_csv_to_mlflow(results_csv=results_csv, round = self.current_round, epoch_per_round = epochs)

        self.best_model = str(trainer.best)
        if not self.best_model.startswith("./"):
            self.best_model = "./" + self.best_model

        self.last_model = str(trainer.last)
        if not self.last_model.startswith("./"):
            self.last_model = "./" + self.last_model
        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                           "message": "Finished round 1!", "round": self.current_round, "best": self.best_model, "last": self.last_model}
        
        # Finish epoch training, send notify to server
        self.send_to_server(notify_data)
        src.Log.print_with_color("[>>>] Finish round 1!", "red")

        # Check training process
        broadcast_queue_name = f'reply_{self.client_id}'
        while True:  # Wait for broadcast
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    return True
                elif received_data["action"] == "CONTINUE":
                    print("Continue training next round")
                    self.current_round += 1
                    args = dict(model=self.last_model,
                                data=dataset_path,
                                epochs=epochs,
                                batch=batch_size,
                                project = f'./runs/detect/{self.client_id}',
                                workers = worker,
                                close_mosaic=0,
                                save_period = valid_epoch_model,
                                optimizer='SGD',
                                lr0=0.01 * (0.92 ** (self.current_round-1)),
                                warmup_epochs=0)
                    trainer = TrainerClass(overrides=args, client_id=self.client_id, layer_id=self.layer_id, num_client=num_client,
                            cut_layer=cut_layer, address=address, username=username, password=password, load_partial_model=load_partial_model, FedAvg=True)
                    trainer.train()

                    # Log to ML FLow
                    save_dir = trainer.save_dir
                    results_csv = save_dir / "results.csv"
                    src.Utils.log_results_csv_to_mlflow(results_csv=results_csv, round = self.current_round, epoch_per_round = epochs)

                    self.best_model = str(trainer.best)
                    if not self.best_model.startswith("./"):
                        self.best_model = "./" + self.best_model

                    self.last_model = str(trainer.last)
                    if not self.last_model.startswith("./"):
                        self.last_model = "./" + self.last_model

                    notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                                    "message": "Finish round 2!", "round": self.current_round, "best": self.best_model, "last": self.last_model}
                    self.send_to_server(notify_data)
            time.sleep(0.5)
                    
    def train_on_device(self, model_path, dataset_path, num_client,cut_layer, num_round, task, epochs, batch_size, worker, address, username, password, load_partial_model, valid_epoch_model):
        self.data_count = 0
        if self.layer_id == 1:

            # Create gradient queue
            forward_queue_name = f'gradient_queue_{self.client_id}'
            self.channel.queue_declare(queue=forward_queue_name, durable=False)
            self.channel.basic_qos(prefetch_count=10)

            result = self.train_on_first_layer(model_path, dataset_path, num_client,cut_layer, num_round, task, epochs, batch_size, worker, address, username, password, load_partial_model, valid_epoch_model)

        elif self.layer_id == 2:
            # Create intermediate queue
            forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'
            self.channel.queue_declare(queue=forward_queue_name, durable=False)
            self.channel.basic_qos(prefetch_count=10)

            # Create label queue
            forward_queue_name = f'label_queue'
            self.channel.queue_declare(queue=forward_queue_name, durable=False)
            self.channel.basic_qos(prefetch_count=10)
            
            result = self.train_on_last_layer(model_path, dataset_path, num_client, cut_layer, num_round, task, epochs, batch_size, worker, address, username, password, load_partial_model, valid_epoch_model)

        if self.event_time:
            src.Log.print_with_color(f"Training time events {self.time_event}", "yellow")
        return result, self.best_model

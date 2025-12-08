import time
import pickle
import pika
from tqdm import tqdm
from engine.yolo.train import Split_Learning_DetectionTrainer, Split_Learning_SegmentationTrainer, Split_Learning_ClassificationTrainer
import src.Log
from ultralytics import YOLO
import torch

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
        self.current_round = 1
    

    def send_to_server(self, message):
        print (f"[>>>] Client {self.client_id} send message to server: {message}")
        self.channel.basic_publish(exchange='',
                                   routing_key='Server_queue',
                                   body=pickle.dumps(message))

    def train_on_first_layer(self, model_path, dataset_path, num_client, cut_layer, num_round, task, epochs, batch_size, worker, address = None, username = None, password = None, load_partial_model=False, valid_epoch_model = -1):
        src.Log.print_with_color(f"--- START TRAINING FIRST LAYER --- CURREN ROUND: {self.current_round} ---", "green")

        from engine.yolo.predict import Split_Learning_DetectionPredictor
        import time

        args = dict(model="fedavg_model_layer_1.pt", source="./bus.jpg")
        predictor = Split_Learning_DetectionPredictor(overrides=args, layer_id=self.layer_id, address=address, username=username, password=password)
        start_time = time.time()
        predictor.predict_cli()
        end_time = time.time()
        total = end_time - start_time
        print(f"Total processing time: {total:.2f} seconds")

        TRAINER = {
            "detect": Split_Learning_DetectionTrainer,
            "segment": Split_Learning_SegmentationTrainer,
            "classify": Split_Learning_ClassificationTrainer,
        }
        TrainerClass = TRAINER.get(task)
        args = dict(model=model_path,
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
                    trainer_last = trainer.last

                    fed_ckpt = torch.load(fed_model_path, map_location='cpu')
                    if isinstance(fed_ckpt, dict) and 'model' in fed_ckpt:
                        fed_sd = fed_ckpt['model'].state_dict()
                    else:
                        fed_sd = fed_ckpt.state_dict() if hasattr(fed_ckpt, 'state_dict') else fed_ckpt

                    last_model = YOLO(trainer_last)
                    last_sd = last_model.model.state_dict()

                    filtered_sd = {k: v for k, v in fed_sd.items() if k in last_sd and v.shape == last_sd[k].shape}

                    print(f"Loaded {len(filtered_sd)}/{len(fed_sd)} weights (skipped mismatched keys like head).")

                    last_model.model.load_state_dict(filtered_sd, strict=False)

                    last_model.save(trainer_last)

                    print(f"Saved to: {trainer_last}")

                    args = dict(resume=trainer_last,
                                epochs=self.current_round*epochs,
                                batch=batch_size,
                                project = f'./runs/detect/{self.client_id}',
                                workers = worker,
                                save_period = valid_epoch_model)
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
        queue_name = f'label_queue'
        result = True
        self.channel.queue_declare(queue=queue_name, durable=False)
        self.channel.queue_declare(queue="number_batch_queue", durable=False)
        self.channel.basic_qos(prefetch_count=10)
        print('Waiting for intermediate output. To exit press CTRL+C')

        src.Log.print_with_color("--- START TRAINING SECOND LAYER ---", "green")

        from engine.yolo.predict import Split_Learning_DetectionPredictor
        import time
        args = dict(model="yolo11n.pt", imgsz = 640)
        predictor = Split_Learning_DetectionPredictor(overrides=args, layer_id=self.layer_id, address=address, username=username, password=password)
        start_time = time.time()
        predictor.predict_cli()
        end_time = time.time()
        total = end_time - start_time
        print(f"Total processing time: {total:.2f} seconds")

        TRAINER = {
            "detect": Split_Learning_DetectionTrainer,
            "segment": Split_Learning_SegmentationTrainer,
            "classify": Split_Learning_ClassificationTrainer,
        }
        TrainerClass = TRAINER.get(task)
        args = dict(model="./yolo11n.pt",
                    data=dataset_path,
                    epochs=epochs,
                    batch=batch_size,
                    project = './runs/detect',
                    workers = worker,
                    save_period = valid_epoch_model
                    # optimizer='AdamW',
                    )
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
                    args = dict(resume=self.last_model,
                                data=dataset_path,
                                epochs=self.current_round*epochs,
                                batch=batch_size,
                                project = f'./runs/detect/{self.client_id}',
                                workers = worker,
                                close_mosaic=0,
                                save_period = valid_epoch_model)
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

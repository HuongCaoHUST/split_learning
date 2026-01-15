import os
import time
import pickle
import sys
import yaml
import mlflow
from ultralytics.data.utils import check_det_dataset
import src.Log
import src.Utils
from src.Validation import ModelValidator
from src.Utils import calculate_latency
from engine.communication.Communication import RabbitMQConnection
from engine.communication.NotificationService import NotificationService
import random

class Server:
    def __init__(self, config_dir):
        with open(config_dir, 'r') as file:
            config = yaml.safe_load(file)

        # RabbitMQ setup
        self.address = config["rabbit"]["address"]
        self.username = config["rabbit"]["username"]
        self.password = config["rabbit"]["password"]
        RabbitMQConnection.delete_old_queues(self.address, self.username, self.password)
        self.rabbitmq_conn = RabbitMQConnection(self.address, self.username, self.password)
        self.rabbitmq_conn.connect()
        self.notification_service = NotificationService(self.rabbitmq_conn)

        # Clients
        self.total_clients = config["server"]["clients"]
        self.num_round = config["server"]["num-round"]
        self.batch_size = config["learning"]["batch-size"]
        self.lr = config["learning"]["learning-rate"]
        self.momentum = config["learning"]["momentum"]
        self.epochs = config["learning"]["epochs"]
        self.worker = config["learning"]["worker"]

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []
        self.count = [0, 0]
        self.last_model_layer_1 = []
        self.last_model_layer_2 = []

        # Model
        self.task = config["model"]["task"]
        self.load_partial_model = config["model"]["load_partial_model"]
        if self.load_partial_model:
            self.model_path = config["model"]["model_path_partial"]
        else:
            self.model_path = config["model"]["model_path"]
        self.cut_layer = config["model"]["cut_layer"]
        if len(self.total_clients) > len(self.cut_layer):
            self.cut_layer = [self.cut_layer[0] for _ in range(len(self.total_clients))]
        self.valid_epoch_model = 1 if config["model"]["valid_epoch_model"] else -1
        self.hybrid_training = config["model"]["hybrid_training"]
        self.output_model = config["model"]["output_model"]

        #Dataset
        self.dataset_path = config["dataset"]["dataset_path"] if not config["dataset"]["iid_datasets"] else self.random_dataset(num_clients=self.total_clients[0])
        print("Dataset paths for clients:", self.dataset_path)

        if len(self.total_clients) > len(self.dataset_path):
            self.dataset_path = [self.dataset_path[0] for _ in range(len(self.total_clients))]

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

        self.rabbitmq_conn.declare_queue('Server_queue')
        self.rabbitmq_conn.consume('Server_queue', self.on_request)

        self.logger = src.Log.Logger(f"{log_path}/app.log")
        filename = os.path.basename(config_dir)
        self.logger.log_info(f"Start Training - File config: {filename}")
        src.Utils.init_csv(f"{log_path}/log/log_validation.csv", headers=["epoch", "precision", "recall", "mAP50", "mAP50-95"])

        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

        # MLflow setup
        mlflow.set_tracking_uri("http://14.225.254.18:5000")
        mlflow.set_experiment("Split_Learning")

    def start(self):
        self.rabbitmq_conn.start_consuming()

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
        action = message.get("action")
        
        handlers = {
            "REGISTER": self._handle_register,
            "NOTIFY": self._handle_notify,
            "VAL_INTER": self._handle_val_inter,
            "UPDATE": self._handle_update,
        }
        
        handler = handlers.get(action)
        if handler:
            handler(message)
        else:
            src.Log.print_with_color(f"Unknown action: {action}", "yellow")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _handle_register(self, message):
        client_id = message["client_id"]
        layer_id = message["layer_id"]

        if (str(client_id), layer_id) not in self.list_clients:
            self.list_clients.append((str(client_id), layer_id))

        src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
        self.register_clients[layer_id - 1] += 1
        if self.register_clients == self.total_clients:
            src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
            self.active_run = mlflow.start_run(run_name="Split Training")
            self.notify_to_clients(run_id=self.active_run.info.run_id)

    def _handle_notify(self, message):
        layer_id = message["layer_id"]
        src.Log.print_with_color(f"[<<<] Received message from client 1: {message}", "blue")
        if layer_id == 1:
            print("BEST MODEL FROM CLIENT:", message["best"])
            print("LAST MODEL FROM CLIENT:", message["last"])
            if message.get("round") == 1:
                self.last_model_layer_1.append(message["last"])
            self.count[0] += 1
        elif layer_id == 2:
            print("BEST MODEL FROM CLIENT:", message["best"])
            print("LAST MODEL FROM CLIENT:", message["last"])
            print("[CHECK] ROUND: ", message.get("round"))
            if message.get("round") == 1:
                self.last_model_layer_2.append(message["last"])
            self.count[1] += 1
            print("COUNT:", self.count)
            print("Received all parameter clients")
            print("LAST MODEL LAYER 1:", self.last_model_layer_1)
            print("LAST MODEL LAYER 2:", self.last_model_layer_2)
            if message.get("round") < self.num_round:
                avg_model_path =self.val_function.average_yolo_models(self.last_model_layer_1, "./fedavg_model_layer_1.pt")
                response_message = {"action": "CONTINUE",
                    "message": "Continue training!",
                    "model_path": avg_model_path}
                self.notification_service.notify_to_all_clients(self.list_clients, response_message)
            else:
                response_message = {"action": "PAUSE",
                    "message": "Pause training and please send your parameters",
                    "parameters": None}
                src.Log.print_with_color(f"[>>>] Sent stop training request to all clients", "red")
                self.notification_service.notify_to_all_clients(self.list_clients, response_message)

    def _handle_val_inter(self, message):
        src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
        layer_id = message["layer_id"]
        layer_map = {
            1: self.val_function.epoch_model_layer_1,
            2: self.val_function.epoch_model_layer_2
        }
        if layer_id in layer_map:
            layer_map[layer_id].append(message["epoch_intermediate"])

    def _handle_update(self, message):
        layer_id = message["layer_id"]
        virtual_machine = message["vm"]
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
            self.logger.log_info(f"Done training - {best}")
            print("BEST_2.pt:", self.val_function.best_model_2)
            mlflow.end_run()
            calculate_latency()
            sys.exit()

    def notify_to_clients(self, status="start", run_id=None):
        src.Log.print_with_color(f"notify_client", "red")
        print("self.list_client: ", self.list_clients)

        dataset_index = 0
        for (client_id, layer_id) in self.list_clients:
            response = {}
            if status == "start":
                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "num_client": self.total_clients,
                            "num_round": self.num_round,
                            "task": self.task,
                            "epochs": self.epochs,
                            "batch_size": self.batch_size,
                            "lr": self.lr,
                            "momentum": self.momentum,
                            "worker": self.worker,
                            "load_partial_model": self.load_partial_model,
                            "valid_epoch_model": self.valid_epoch_model,
                            "run_id": run_id}
                
                if layer_id == 1:
                    response["model_path"] = self.model_path[0]
                    response["cut_layer"] = self.cut_layer[dataset_index]
                    response["dataset_path"] = self.dataset_path[dataset_index]
                    if self.concatenate_datasets and dataset_index != 0:
                        delta_nc = self.nc_list_cumulative[dataset_index - 1]
                        response["concatenate_datasets"] = True
                        response["delta_nc"] = delta_nc
                    dataset_index += 1
                elif layer_id == 2:
                    response["model_path"] = self.model_path[0]
                    response["cut_layer"] = self.cut_layer[0]
                    response["dataset_path"] = self.dataset_path[0]
                
                self.time_start = time.time_ns()
                self.notification_service.send_to_client(client_id, response)
            
            if status == "continue":
                response = {"action": "CONTINUE"}
                if layer_id == 1:
                    self.notification_service.send_to_client(client_id, response)

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
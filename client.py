import pika
import uuid
import argparse
import yaml

import torch
import time
import src.Log
from src.Client import Client
from src.Trainning import Trainning
import threading
import psutil
import requests


parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')
parser.add_argument('--docker', action='store_true', help='Run inside Docker container')
parser.add_argument('--vm', action='store_true', help='Run inside virtual machine')
parser.add_argument('--event_time', type=bool, default=False, required=False, help='Log event time for debug mode')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


client_id = uuid.uuid4()
if args.layer_id == 2:
    address = "172.18.0.2"
elif args.layer_id == 1 and args.docker == True and args.vm == False:
    address = "172.18.0.2"
elif args.layer_id == 1 and args.docker == True and args.vm == True:
    address = "192.168.0.101"    
else:
    address = "127.0.0.1"
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

device = None

if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")

def connection(username, password, address):
    credentials = pika.PlainCredentials(username, password)
    while True:
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(address, 5672, '/', credentials)
            )
            return connection
        except pika.exceptions.AMQPConnectionError as e:
            time.sleep(1)

PUSHGATEWAY = "http://14.225.254.18:9091/metrics/job/pi_metrics/instance/pi1"
def push_metrics_loop():
    while True:
        try:
            cpu = psutil.cpu_percent() / 100
            ram_used = psutil.virtual_memory().used / (1024 * 1024)
            disk = psutil.disk_usage('/').percent / 100

            data = f"""
            # HELP pi_cpu_usage CPU usage
            # TYPE pi_cpu_usage gauge
            pi_cpu_usage {cpu}
            # HELP pi_ram_usage RAM usage
            # TYPE pi_ram_usage gauge
            pi_ram_usage {ram_used}
            # HELP pi_disk_usage Disk usage
            # TYPE pi_disk_usage gauge
            pi_disk_usage {disk}
            """
            requests.post(PUSHGATEWAY, data=data)
        except Exception as e:
            print("Failed to push metrics:", e)
        time.sleep(10)

if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id, "docker": args.docker, "virtual machine": args.vm}
    connection = connection(username, password, address)
    channel = connection.channel()
    
    metrics_thread = threading.Thread(target=push_metrics_loop, daemon=True)
    metrics_thread.start()

    trainning = Trainning(client_id, args.layer_id, channel, device, args.event_time)
    client = Client(client_id, args.layer_id, address, username, password, trainning.train_on_device, device, args.vm)
    client.send_to_server(data)
    time.sleep(8)
    client.wait_response()
import os
import pandas as pd
import yaml
import pika
from pathlib import Path

IMG_FORMATS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}  # image formats

def read_file(file_path):
        with open(file_path, "rb") as file:
            return file.read()

def init_csv(csv_file, headers):
        """
        Init csv log file
        """
        df = pd.DataFrame(columns=headers)
        df.to_csv(csv_file, index=False)

def log_to_csv(csv_file, data_dict):
        """
        Log to csv file
        """
        df = pd.DataFrame([data_dict])
        df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

def save_model_file(best_model, best_dir="./best_model_vm"):
        save_dir = os.path.abspath(best_dir)
        os.makedirs(save_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("best_") and f.endswith(".pt")]
        indices = []

        for f in existing_files:
            try:
                index = int(f.replace("best_", "").replace(".pt", ""))
                indices.append(index)
            except ValueError:
                pass

        next_index = max(indices, default=0) + 1
        filename = f"best_{next_index}.pt"
        file_path = os.path.join(save_dir, filename)

        with open(file_path, "wb") as f:
            f.write(best_model)

        return file_path

def check_dataset(dataset_paths, batch_size):
    nb_distributed = []
    for dataset_path in dataset_paths:
        with open(dataset_path, "r") as f:
            data = yaml.safe_load(f)

        raw_path = data["path"]
        dataset_info = Path(raw_path).resolve()
        train_dataset = dataset_info / "train/images"
        if not train_dataset.exists():
            print(f"Thư mục {train_dataset} không tồn tại")
            continue
        image_files = [p for p in train_dataset.rglob("*") if p.suffix.lower() in IMG_FORMATS]
        nb = calculate_nb(len(image_files), batch_size)
        nb_distributed.append(nb)
    return nb_distributed

def calculate_nb(number_images, batch_size):
    """
    Calculate the number of batches
    """
    return (number_images // batch_size)

def connect_rabbitmq(address, username, password):
        try:
            credentials = pika.PlainCredentials(username, password)
            parameters = pika.ConnectionParameters(
                host=address,
                port=5672,
                virtual_host='/',
                credentials=credentials
            )
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            return channel
        except Exception as e:
            return None
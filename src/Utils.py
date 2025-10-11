import os
import pandas as pd
import yaml
import pika
from pathlib import Path
import shutil

IMG_FORMATS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}  # image formats

def read_file(file_path):
        with open(file_path, "rb") as file:
            return file.read()

def init_csv(csv_file, headers):
        """
        Init csv log file
        """
        folder = os.path.dirname(csv_file)
        if folder:
            os.makedirs(folder, exist_ok=True)
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
        
import shutil
import yaml
from pathlib import Path

def split_dataset(yaml_path, num_client=5):
    yaml_paths = [] 
    src_root = Path(yaml_path).parent

    src_train_images = src_root / "train" / "images"
    src_train_labels = src_root / "train" / "labels"
    src_val_images = src_root / "valid" / "images"
    src_val_labels = src_root / "valid" / "labels"

    dst_base = Path("./datasets/clients")
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    # Lấy danh sách ảnh train
    images = sorted([f for f in src_train_images.iterdir() if f.is_file()])
    images_per_client = len(images) // num_client

    for i in range(num_client):
        client_dir = dst_base / f"client{i+1}"
        train_img_dir = client_dir / "train" / "images"
        train_lbl_dir = client_dir / "train" / "labels"
        val_img_dir = client_dir / "valid" / "images"
        val_lbl_dir = client_dir / "valid" / "labels"

        # Tạo thư mục cần thiết
        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_lbl_dir.mkdir(parents=True, exist_ok=True)
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)

        start = i * images_per_client
        end = (i + 1) * images_per_client
        client_images = images[start:end]

        for img in client_images:
            shutil.copy(img, train_img_dir)
            label_file = src_train_labels / (img.stem + ".txt")
            if label_file.exists():
                shutil.copy(label_file, train_lbl_dir)
        for img in src_val_images.iterdir():
            if img.is_file():
                shutil.copy(img, val_img_dir)
                label_file = src_val_labels / (img.stem + ".txt")
                if label_file.exists():
                    shutil.copy(label_file, val_lbl_dir)
        client_yaml = yaml_data.copy()
        client_yaml["train"] = "../train/images"
        client_yaml["val"] = "../valid/images"
        client_yaml["test"] = "../test/images"

        out_yaml = client_dir / "data.yaml"
        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.dump(client_yaml, f, sort_keys=False, allow_unicode=True)
        rel_path = Path("./datasets") / out_yaml.relative_to(Path("./datasets"))
        yaml_paths.append(str(rel_path))
        print(f"✅ Client {i+1}: {len(client_images)} ảnh train + {len(list(src_val_images.iterdir()))} ảnh val + file yaml")

    return yaml_paths
import os
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image
import shutil
import random

def build_splits_mnist_cls(num_clients=10, alpha=0.5, out_root="mnist_yolo_cls", seed=42, val_size=100):
    random.seed(seed)
    np.random.seed(seed)
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    n_classes = 10
    data_indices = [[] for _ in range(n_classes)]
    for idx, (_, label) in enumerate(mnist_train):
        data_indices[label].append(idx)
    client_indices = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        np.random.shuffle(data_indices[c])
        proportions = np.random.dirichlet(alpha=[alpha]*num_clients)
        proportions = (np.cumsum(proportions) * len(data_indices[c])).astype(int)[:-1]
        split = np.split(data_indices[c], proportions)
        for i in range(num_clients):
            client_indices[i].extend(split[i])
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)
    for cid in range(num_clients):
        indices = client_indices[cid]
        random.shuffle(indices)

        # Chia train/val
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        for split_name, split_indices in [("train", train_idx), ("val", val_idx)]:
            for idx in tqdm(split_indices, desc=f"Client {cid} {split_name}"):
                img, label = mnist_train[idx]
                img = (img.squeeze().numpy()*255).astype(np.uint8)
                class_dir = os.path.join(out_root, f"client_{cid}", split_name, str(label))
                os.makedirs(class_dir, exist_ok=True)
                img_path = os.path.join(class_dir, f"{idx}.png")
                Image.fromarray(img).save(img_path)

    print(f"✅ Dataset saved to {out_root}, with {num_clients} clients (alpha={alpha}), val_size={val_size}")
build_splits_mnist_cls(num_clients=200, alpha=1, out_root="mnist_yolo_cls_dirichlet", seed=42, val_size=100)

import os, math, random, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def split_dirichlet_per_class(indices, num_clients, alpha, rng):
    if len(indices) == 0:
        return [[] for _ in range(num_clients)]
    rng.shuffle(indices)
    props = rng.dirichlet([alpha] * num_clients)
    counts = [int(round(p * len(indices))) for p in props]
    diff = len(indices) - sum(counts)
    for i in np.argsort([-p for p in props])[:abs(diff)]:
        counts[i] += 1 if diff > 0 else -1
    out, start = [], 0
    for c in counts:
        out.append(indices[start:start+c])
        start += c
    return out

def build_splits_cifar10(num_clients=5, alpha=0.5, val_ratio=0.1, seed=42,
                         out_root="cifar10_yolo_cls_dirichlet"):
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    ds = CIFAR10(root="./data", train=True, download=True)
    targets = np.array(ds.targets)
    class_to_indices = {c: np.where(targets == c)[0].tolist() for c in range(10)}
    client_indices = [[] for _ in range(num_clients)]
    for c in range(10):
        per_client = split_dirichlet_per_class(class_to_indices[c], num_clients, alpha, rng_np)
        for k in range(num_clients):
            client_indices[k].extend(per_client[k])
    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    def write_image(pil_img, dst_path):
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(dst_path, format="JPEG", quality=95)
    for k in range(num_clients):
        idxs = client_indices[k]
        rng_py.shuffle(idxs)

        n_val = int(round(val_ratio * len(idxs)))
        val_set = set(idxs[:n_val])
        train_set = set(idxs[n_val:])

        client_dir = out_root / f"client_{k}"
        for split_name, split_set in [("train", train_set), ("val", val_set)]:
            for i in split_set:
                img, label = ds[i]
                cls_name = CLASS_NAMES[label]
                dst = client_dir / split_name / cls_name / f"{i}.jpg"
                write_image(img, dst)
        yaml_text = f"""# Auto-generated
path: {str(client_dir).replace('\\', '/')}
train: train
val: val
names:
  0: airplane
  1: automobile
  2: bird
  3: cat
  4: deer
  5: dog
  6: frog
  7: horse
  8: ship
  9: truck
"""
        (out_root / f"client_{k}.yaml").write_text(yaml_text, encoding="utf-8")
    summary_lines = []
    for k in range(num_clients):
        client_dir = out_root / f"client_{k}"
        for split_name in ["train", "val"]:
            counts = []
            total = 0
            for ci, cname in enumerate(CLASS_NAMES):
                d = client_dir / split_name / cname
                n = len(list(d.glob("*.jpg"))) if d.exists() else 0
                counts.append(n)
                total += n
            summary_lines.append(
                f"client_{k}/{split_name}: total={total}, per_class={counts}"
            )
    (out_root / "SPLIT_SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Done. Root: {out_root.resolve()}")
    return out_root

if __name__ == "__main__":
    # Eg: 10 client, alpha=0.3 (Non-IID mạnh), val 10%
    build_splits_cifar10(num_clients=100, alpha=0.1, val_ratio=0.1, seed=42,
                         out_root="cifar10_yolo_cls_dirichlet")

# dirichlet_cifar10_yolo_cls.py
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
    """
    Chia danh sách index của 1 class cho K client theo tỉ lệ Dirichlet(alpha).
    Trả về list[list[int]]: indices cho mỗi client.
    """
    if len(indices) == 0:
        return [[] for _ in range(num_clients)]
    rng.shuffle(indices)
    props = rng.dirichlet([alpha] * num_clients)
    # chuyển tỉ lệ thành số lượng mẫu nguyên
    counts = [int(round(p * len(indices))) for p in props]
    # hiệu chỉnh để tổng đúng bằng len(indices)
    diff = len(indices) - sum(counts)
    # phân bổ chênh lệch (có thể âm hoặc dương)
    for i in np.argsort([-p for p in props])[:abs(diff)]:
        counts[i] += 1 if diff > 0 else -1
    # cắt mảng
    out, start = [], 0
    for c in counts:
        out.append(indices[start:start+c])
        start += c
    return out

def build_splits_cifar10(num_clients=5, alpha=0.5, val_ratio=0.1, seed=42,
                         out_root="cifar10_yolo_cls_dirichlet"):
    """
    Tạo dataset cho YOLO classification.
    Cấu trúc:
      out_root/
        client_0/
          train/<class>/*.jpg
          val/<class>/*.jpg
        client_1/...
      + mỗi client có 1 YAML để train: client_0.yaml ...
    """
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    # 1) Tải CIFAR-10 (train 50k ảnh). Có thể thêm test nếu muốn.
    ds = CIFAR10(root="./data", train=True, download=True)
    targets = np.array(ds.targets)

    # 2) Gom index theo class
    class_to_indices = {c: np.where(targets == c)[0].tolist() for c in range(10)}

    # 3) Chia theo Dirichlet cho từng class
    client_indices = [[] for _ in range(num_clients)]
    for c in range(10):
        per_client = split_dirichlet_per_class(class_to_indices[c], num_clients, alpha, rng_np)
        for k in range(num_clients):
            client_indices[k].extend(per_client[k])

    # 4) Tách train/val cho từng client theo val_ratio
    #    (giữ non-overlap giữa các client)
    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Hàm tiện ích ghi ảnh ra cấu trúc YOLO-Cls
    def write_image(pil_img, dst_path):
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(dst_path, format="JPEG", quality=95)

    # 5) Ghi ảnh ra đĩa theo từng client
    for k in range(num_clients):
        idxs = client_indices[k]
        rng_py.shuffle(idxs)

        n_val = int(round(val_ratio * len(idxs)))
        val_set = set(idxs[:n_val])
        train_set = set(idxs[n_val:])

        client_dir = out_root / f"client_{k}"
        for split_name, split_set in [("train", train_set), ("val", val_set)]:
            for i in split_set:
                img, label = ds[i]             # img: PIL, label: 0..9
                cls_name = CLASS_NAMES[label]
                dst = client_dir / split_name / cls_name / f"{i}.jpg"
                write_image(img, dst)

        # 6) Tạo YAML cho YOLO-cls
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

    # 7) Ghi meta thông tin split để kiểm tra phân phối
    #    (đếm số ảnh mỗi class trong train/val của từng client)
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
    build_splits_cifar10(num_clients=4, alpha=0.1, val_ratio=0.1, seed=42,
                         out_root="cifar10_yolo_cls_dirichlet_a0_3_K10")

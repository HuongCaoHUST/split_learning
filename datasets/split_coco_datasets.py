import numpy as np
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from ultralytics.utils.downloads import download

# ================== HELPER FUNCTION ==================
def read_yolo_classes(label_file):
    classes = []
    with open(label_file) as f:
        for line in f:
            if line.strip():
                classes.append(int(line.split()[0]))
    return classes

# ================== 1. DOWNLOAD DATA ==================
def download_dataset():
    print("\n⬇️  STARTING COCO DATASET DOWNLOAD...")
    
    base_dir = Path("./") 
    
    # Download labels
    segments = False
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    urls = [url + ("coco2017labels-segments.zip" if segments else "coco2017labels.zip")]
    download(urls, dir=base_dir) 

    # Download images
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
    ]
    download(urls, dir=base_dir / "images", threads=3)
    print("✅ Download complete.\n")

# ================== 2. PROCESS & SPLIT ==================
def process_and_split_data():
    print("🔄 STARTING DATA SPLIT FOR CLIENTS...")

    # Config
    label_dir = Path("./coco/labels/train2017")
    image_dir = Path("./images/train2017")
    output_root = Path("./COCO_clients")

    valid_images_src = Path(r"./images/val2017")
    valid_labels_src = Path(r"./coco/labels/val2017")

    NUM_CLIENTS = 8
    ALPHA = 10000

    if not label_dir.exists() or not image_dir.exists():
        print(f"❌ Error: Source directories not found ({label_dir} or {image_dir}).")
        return

    # 1. Collect data by class
    class_to_images = defaultdict(list)
    all_labels = list(label_dir.glob("*.txt"))

    print(f"\n🚀 Processing {len(all_labels)} label files...")
    for lbl in tqdm(all_labels, desc="[1/4] Reading Labels"):
        classes = read_yolo_classes(lbl)
        for cls in classes:
            class_to_images[cls].append(lbl)

    # 2. Dirichlet split
    client_files = [set() for _ in range(NUM_CLIENTS)]
    assigned_labels = set()

    for cls, files in tqdm(class_to_images.items(), desc="[2/4] Dirichlet Split"):
        files = list(set(files))
        random.shuffle(files)

        proportions = np.random.dirichlet([ALPHA] * NUM_CLIENTS)
        split_points = (np.cumsum(proportions) * len(files)).astype(int)[:-1]
        splits = np.split(files, split_points)

        for i in range(NUM_CLIENTS):
            for f in splits[i]:
                if f not in assigned_labels:
                    client_files[i].add(f)
                    assigned_labels.add(f)

    # Create directories
    for i in range(NUM_CLIENTS):
        (output_root / f"COCO{i+1}/images/train").mkdir(parents=True, exist_ok=True)
        (output_root / f"COCO{i+1}/labels/train").mkdir(parents=True, exist_ok=True)
        (output_root / f"COCO{i+1}/images/valid").mkdir(parents=True, exist_ok=True)
        (output_root / f"COCO{i+1}/labels/valid").mkdir(parents=True, exist_ok=True)

    # 3. Copy Train files
    print("\n📂 Copying Train data (This may take time)...")
    for i, files in enumerate(client_files):
        for lbl in tqdm(files, desc=f"[3/4] Client {i+1} Copying", total=len(files), leave=False):
            img = image_dir / (lbl.stem + ".jpg")
            shutil.copy(lbl, output_root / f"COCO{i+1}/labels/train" / lbl.name)
            if img.exists():
                shutil.copy(img, output_root / f"COCO{i+1}/images/train" / img.name)

    # 4. Copy Valid files
    print("\n📂 Copying Validation data...")
    all_valid_imgs = list(valid_images_src.glob("*"))
    all_valid_lbls = list(valid_labels_src.glob("*"))

    for i in range(NUM_CLIENTS):
        target_img_dir = output_root / f"COCO{i+1}/images/valid"
        target_lbl_dir = output_root / f"COCO{i+1}/labels/valid"
        
        if i == 0:
            # Client 1 gets full validation set
            for valid_img in tqdm(all_valid_imgs, desc="Client 1 Valid Images", leave=False):
                shutil.copy(valid_img, target_img_dir / valid_img.name)
            for valid_lbl in tqdm(all_valid_lbls, desc="Client 1 Valid Labels", leave=False):
                shutil.copy(valid_lbl, target_lbl_dir / valid_lbl.name)
        else:
            # Others get a small subset
            for valid_img in sorted(all_valid_imgs)[:5]:
                shutil.copy(valid_img, target_img_dir / valid_img.name)
            for valid_lbl in sorted(all_valid_lbls)[:5]:
                shutil.copy(valid_lbl, target_lbl_dir / valid_lbl.name)

    print("✅ Validation copy complete.")

    # 5. Statistics
    print(f"\n📊 RESULTS FOR alpha = {ALPHA}")
    print("Calculating statistics...")

    for i, files in enumerate(client_files):
        counter = Counter()
        for lbl in files:
            classes = read_yolo_classes(lbl)
            counter.update(classes)

        print(f"\nClient {i+1}:")
        for cls in sorted(counter):
            print(f"  Class {cls}: {counter[cls]} bbox")

# ================== MAIN ==================
if __name__ == "__main__":
    download_dataset()
    process_and_split_data()
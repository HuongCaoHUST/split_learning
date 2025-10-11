import os
import json
import pandas as pd

# Thư mục input (chứa file JSON)
input_dir = "./train_10_10/new"

# Thư mục output (chứa file CSV)
output_dir = "./train_10_10/new/file_csv"
os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

# Duyệt toàn bộ file JSON trong thư mục input
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".json", ".csv"))

        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Lỗi JSON trong file {filename}: {line}")

        if not records:
            print(f"⚠️ Bỏ qua {filename} (không có dữ liệu hợp lệ)")
            continue

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"✅ Đã chuyển {filename} → {output_path}")

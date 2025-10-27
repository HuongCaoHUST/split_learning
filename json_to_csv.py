import pandas as pd
import json

# Đọc file json lines
input_file = "./docker_stats_8.json"
output_file = "./docker_stats_8.csv"

# Đọc từng dòng và parse json
records = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# Chuyển thành DataFrame
df = pd.DataFrame(records)

# Xuất ra CSV
df.to_csv(output_file, index=False)

print(f"Đã lưu thành CSV: {output_file}")

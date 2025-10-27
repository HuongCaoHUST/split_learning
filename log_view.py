import os
import json
import time
from datetime import datetime

def get_next_log_filename(base_name, ext="json"):
    i = 1
    while os.path.exists(f"{base_name}_{i}.{ext}"):
        i += 1
    return f"{base_name}_{i}.{ext}"

DOCKER_LOG_FILE = get_next_log_filename("docker_stats")
GPU_PROC_LOG_FILE = get_next_log_filename("gpu_stats")

def log_docker_stats():
    output = os.popen('docker stats --no-stream --format "{{json .}}"').read().strip().splitlines()
    stats_list = []
    for line in output:
        try:
            data = json.loads(line)
            data["timestamp"] = datetime.now().isoformat()
            stats_list.append(data)
        except json.JSONDecodeError:
            continue

    with open(DOCKER_LOG_FILE, "a", encoding="utf-8") as f:
        for stat in stats_list:
            f.write(json.dumps(stat) + "\n")
    print(f"[{datetime.now().isoformat()}] Logged {len(stats_list)} Docker containers -> {DOCKER_LOG_FILE}")

def log_gpu_processes():
    gpu_output = os.popen(
        'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits'
    ).read().strip().splitlines()

    proc_stats = []
    for line in gpu_output:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            proc_data = {
                "gpu_uuid": parts[0],
                "pid": parts[1],
                "process_name": parts[2],
                "used_memory": parts[3] + " MiB",
                "timestamp": datetime.now().isoformat()
            }
            proc_stats.append(proc_data)

    with open(GPU_PROC_LOG_FILE, "a", encoding="utf-8") as f:
        for stat in proc_stats:
            f.write(json.dumps(stat) + "\n")

    print(f"[{datetime.now().isoformat()}] Logged {len(proc_stats)} GPU processes -> {GPU_PROC_LOG_FILE}")

if __name__ == "__main__":
    while True:
        log_docker_stats()
        # log_gpu_processes()
        time.sleep(0.5)

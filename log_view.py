import os
import json
import time
from datetime import datetime

LOG_FILE = "./docker_stats.json"

while True:
    output = os.popen('docker stats --no-stream --format "{{json .}}"').read().strip().splitlines()
    
    stats_list = []
    for line in output:
        data = json.loads(line)
        data["timestamp"] = datetime.now().isoformat()
        stats_list.append(data)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for stat in stats_list:
            f.write(json.dumps(stat) + "\n")

    print(f"[{datetime.now().isoformat()}] Logged {len(stats_list)} containers")
    time.sleep(1)

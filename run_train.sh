#!/bin/bash
set -e

docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d

python3 log_view.py &
LOG_PID=$!

sleep 30
python3 get_container_pid.py &

echo "Press 'q' to stop log_view.py..."
while true; do
    read -n 1 key
    if [[ $key == "q" ]]; then
        kill $LOG_PID
        wait $LOG_PID 2>/dev/null
        break
    fi
done

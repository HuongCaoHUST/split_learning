#!/bin/bash
set -e

python3 log_view.py &
LOG_PID=$!
docker compose -f docker-compose1.yaml up --abort-on-container-exit --exit-code-from server --scale client_1=2
python3 get_container_pid.py
kill $LOG_PID
docker compose down

python3 log_view.py &
LOG_PID=$!
docker compose -f docker-compose2.yaml up --abort-on-container-exit --exit-code-from server --scale client_1=2
python3 get_container_pid.py
kill $LOG_PID
docker compose down




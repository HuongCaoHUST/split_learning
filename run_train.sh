#!/bin/bash
set -e

docker compose up -d
python3 log_view.py &
sleep 30
python3 get_container_pid.py

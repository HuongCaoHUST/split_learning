@echo off
echo Starting containers...
docker compose -f docker-compose1.yaml up --abort-on-container-exit --exit-code-from server
docker compose wait
docker compose down

echo Starting containers...
docker compose -f docker-compose1.yaml up --abort-on-container-exit --exit-code-from server
docker compose wait
docker compose down



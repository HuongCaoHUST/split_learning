import subprocess
import json
import datetime

def get_running_containers():
    result = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True, check=True)
    return [cid for cid in result.stdout.strip().split('\n') if cid]

def get_container_pid(container_id):
    try:
        result = subprocess.run(['docker', 'inspect', container_id], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return data[0]['State']['Pid']
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return None

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"container_pids_{timestamp}.txt"
    container_ids = get_running_containers()
    
    with open(output_file, 'w') as f:
        f.write("Container ID\tPID\n")
        f.write("-" * 50 + "\n")
        for cid in container_ids:
            pid = get_container_pid(cid)
            f.write(f"{cid}\t{pid if pid is not None else 'Khong lay duoc PID'}\n")

if __name__ == "__main__":
    main()
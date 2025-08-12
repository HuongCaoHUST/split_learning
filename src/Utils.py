import os
import pandas as pd

def init_csv(csv_file, headers):
        """
        Init csv log file
        """
        df = pd.DataFrame(columns=headers)
        df.to_csv(csv_file, index=False)

def log_to_csv(csv_file, data_dict):
        """
        Log to csv file
        """
        df = pd.DataFrame([data_dict])
        df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

def save_model_file(best_model, best_dir="./best_model_vm"):
        save_dir = os.path.abspath(best_dir)
        os.makedirs(save_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("best_") and f.endswith(".pt")]
        indices = []

        for f in existing_files:
            try:
                index = int(f.replace("best_", "").replace(".pt", ""))
                indices.append(index)
            except ValueError:
                pass

        next_index = max(indices, default=0) + 1
        filename = f"best_{next_index}.pt"
        file_path = os.path.join(save_dir, filename)

        with open(file_path, "wb") as f:
            f.write(best_model)

        return file_path
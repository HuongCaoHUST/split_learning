import os
import pandas as pd

def change_state_dict(state_dicts, i):
    def change_name(name):
        parts = name.split(".", 1)
        number = int(parts[0]) + i
        name = f"{number}" + "." + parts[1]
        return name
    new_state_dict = {}
    for key, value in state_dicts.items():
        new_key = change_name(key)
        new_state_dict[new_key] = value
    return new_state_dict

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

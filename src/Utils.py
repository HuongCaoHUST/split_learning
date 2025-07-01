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


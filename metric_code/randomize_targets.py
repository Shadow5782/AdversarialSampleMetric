import os
import random

data_path = "2_Code/data/targeted_data"

file_names = os.listdir(data_path)

for i in range(0, len(file_names)):
    name = file_names[i].split("_")
    ending = name[5].split(".")
    new_target = int(ending[0])
    while True:
        new_target = random.randint(0,9)
        if new_target != int(ending[0]):
            break
    new_name = f"{name[0]}_{name[1]}_{name[2]}_{name[3]}_{name[4]}_{new_target}.png"
    # print(file_names[i], new_name)
    os.rename(data_path + "/" + file_names[i], data_path + "/" + new_name)

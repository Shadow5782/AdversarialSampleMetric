import os

data_path = "2_Code/data/standardNet_subst/greyBoxData"

file_names = os.listdir(data_path)

for i in range(0, len(file_names)):
    new_name = file_names[i].split("_")
    new_name = f"{new_name[0]}_{i}_{new_name[2]}_{new_name[3]}_{new_name[4]}_{new_name[5]}"
    # print(file_names[i], new_name)
    os.rename(data_path + "/" + file_names[i], data_path + "/" + new_name)

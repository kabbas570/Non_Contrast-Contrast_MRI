
folder_path = r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\before'

import os
import shutil

# Path to the folder containing .npy files

# List all .npy files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

# Dictionary to store image and ground truth pairs by their common identifier
file_pairs = {}

# Identify common identifier in filenames and pair image and ground truth files
for file in file_list:
    if '_seg_data.npy' in file:
        identifier = file.split('_seg_data.npy')[0]  # Assuming '_series' identifies the identifier
        img_file = file.replace('_seg_data.npy', '_reg_lge_pixel_data.npy')  # Generating image filename
        
        if img_file in file_list:
            if identifier not in file_pairs:
                file_pairs[identifier] = [(img_file, file)]
            else:
                file_pairs[identifier].append((img_file, file))

# Move files to folders based on the identifier
for identifier, files in file_pairs.items():
    folder_name = f"{identifier}"
    folder_path_new = os.path.join(folder_path, folder_name)
    os.makedirs(folder_path_new, exist_ok=True)
    
    # for img_file, gt_file in files:
    #     for file in [img_file, gt_file]:
    #         src_path = os.path.join(folder_path, file)
    #         dst_path = os.path.join(folder_path_new, file)
    #         shutil.move(src_path, dst_path)
    #         print(f"Moved {file} to {folder_name}.")

    for img_file, gt_file in files:
        for file in [img_file, gt_file]:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(folder_path_new, file)
            # Add '_gt' to the ground truth filename
            if gt_file in file:
                dst_path = dst_path.replace('.npy', '_gt.npy')
            shutil.move(src_path, dst_path)
            print(f"Moved {file} to {folder_name}.")



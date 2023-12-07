import numpy as np
from sklearn.model_selection import KFold
import os
import shutil

# Path to the folder containing .npy files
folder_path = r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\CINE - Copy'

# List all .npy files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

# Number of folds for cross-validation
num_folds = 5

# Calculate approximately equal number of files per fold
files_per_fold = len(file_list) // num_folds

# Create folders for each fold
for i in range(num_folds):
    fold_folder = os.path.join(folder_path, f'fold_{i+1}')
    os.makedirs(fold_folder, exist_ok=True)

# Initialize KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Move files to each fold folder
fold_index = 1
for train_index, test_index in kf.split(file_list):
    fold_files = np.array(file_list)[test_index]
    
    fold_folder = os.path.join(folder_path, f'fold_{fold_index}')
    
    # Move files to respective fold folder
    for file in fold_files:
        src_path = os.path.join(folder_path, file)
        dst_path = os.path.join(fold_folder, file)
        
        try:
            shutil.move(src_path, dst_path)
            print(f"Moved {file} to {fold_folder} for training.")
        except FileNotFoundError:
            print(f"File {file} not found at {src_path}. Skipping...")
    
    fold_index += 1

import os
import numpy as np
import SimpleITK as sitk

# Path to the folder containing .npy files
folder_path = r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\LGE'

# Output folder for .nrrd files
output_folder = r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\LGE\nrrd'  # Replace with your desired output folder path
os.makedirs(output_folder, exist_ok=True)

# List all .npy files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

# Convert all .npy files to .nrrd format
for file in file_list:
    file_path = os.path.join(folder_path, file)
    if file.endswith('.npy'):
        # Load the .npy file
        numpy_array = np.load(file_path,allow_pickle=True)[0][0]
        
        # Create a SimpleITK image
        sitk_image = sitk.GetImageFromArray(numpy_array)
        
        # Define the output file path with the same filename and .nrrd extension
        output_file_path = os.path.join(output_folder, os.path.splitext(file)[0] + '.nrrd')
        
        # Save the SimpleITK image as .nrrd
        sitk.WriteImage(sitk_image, output_file_path)
        print(f"Converted {file} to .nrrd format.")

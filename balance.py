import os
import pandas as pd
import shutil
import numpy as np

# Read the modified CSV file
df = pd.read_csv("/Users/aryansharma/Desktop/ml-project/modified_data.csv")

# Filter RG images with ".jpg" extension
rg_jpg_images = df[(df['Final Label'] == 'RG') & (df['dataextension'] == '.jpg')]['Eye ID'].tolist()

# Filter NRG images with ".jpg" extension
nrg_jpg_images = df[(df['Final Label'] == 'NRG') & (df['dataextension'] == '.jpg')]['Eye ID'].tolist()

# Determine the required number of images for each class
num_images_per_class = 2000  # Total number of images (NRG + RG)
num_images_each_class = num_images_per_class // 2

# Randomly select RG images for oversampling
selected_rg_images = np.random.choice(rg_jpg_images, num_images_each_class, replace=True)

# Create a new folder to store the selected images
output_folder = '/Users/aryansharma/Desktop/ml-project/balanced_images'
os.makedirs(output_folder, exist_ok=True)

# Copy selected RG images to the output folder
for image in selected_rg_images:
    src = os.path.join('/Users/aryansharma/Desktop/ml-project/selected_images', f'{image}.jpg')
    dst = os.path.join(output_folder, f'{image}.jpg')
    shutil.copy(src, dst)

# Copy selected NRG images to the output folder
for image in nrg_jpg_images:
    src = os.path.join('/Users/aryansharma/Desktop/ml-project/selected_images', f'{image}.jpg')
    dst = os.path.join(output_folder, f'{image}.jpg')
    shutil.copy(src, dst)

print("Images copied successfully to the new folder.")

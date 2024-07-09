import pandas as pd
import os
import shutil

# Path to your CSV file
csv_file_path = '/Users/aryansharma/Desktop/ml-project/1.csv'

# Directory containing images
image_directory = '/Users/aryansharma/Desktop/ml-project/combine'

# Directory to copy images to
new_image_directory = '/Users/aryansharma/Desktop/ml-project/selected_images'

try:
    # Read the first 1000 rows from the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, nrows=1000, sep=';')

    # Check if 'Eye ID' column exists in the DataFrame
    if 'Eye ID' not in df.columns:
        raise KeyError("Column 'Eye ID' not found in the DataFrame.")

    # Create the new directory if it doesn't exist
    os.makedirs(new_image_directory, exist_ok=True)

    # Copy images corresponding to the first 1000 rows to the new directory
    for eye_id in df['Eye ID']:
        for ext in ['.jpg', '.jpeg', '.png']:  # Check for common image extensions
            image_path = os.path.join(image_directory, f"{eye_id}{ext}")
            if os.path.exists(image_path):
                shutil.copy(image_path, new_image_directory)
                break

    print("Images copied to new directory successfully.")

except FileNotFoundError:
    print(f"File '{csv_file_path}' not found. Please check the file path and try again.")

except KeyError as e:
    print(f"KeyError: {e}. Please check the column name in your CSV file.")

except Exception as e:
    print(f"An error occurred: {e}")

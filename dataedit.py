import pandas as pd
import os

# Path to your CSV file
csv_file_path = '/Users/aryansharma/Desktop/ml-project/1.csv'

# Read the first 1000 rows from the CSV file into a DataFrame
try:
    df = pd.read_csv(csv_file_path, nrows=1000, sep=';')
    print("Original DataFrame:")
    print(df.head())  # Print the first few rows of the DataFrame

    # Directory containing images
    image_directory = '/Users/aryansharma/Desktop/ml-project/combine'

    # Check if 'Eye ID' column exists in the DataFrame
    if 'Eye ID' not in df.columns:
        raise KeyError("Column 'Eye ID' not found in the DataFrame.")

    # Define a function to get the extension of an image file
    def get_extension(eye_id):
        for ext in ['.jpg', '.jpeg', '.png']:  # Check for common image extensions
            image_path = os.path.join(image_directory, f"{eye_id}{ext}")
            if os.path.exists(image_path):
                return ext
        return None

    # Extract the extensions for the first 1000 rows
    extensions = []
    for eye_id in df['Eye ID']:
        extensions.append(get_extension(eye_id))

    # Add the extensions as a new column 'dataextension' to the DataFrame
    df['dataextension'] = extensions

    # Print the modified DataFrame
    print("\nModified DataFrame:")
    print(df)

    # Write the modified DataFrame with the added column to a new CSV file
    output_csv_file_path = '/Users/aryansharma/Desktop/ml-project/modified_data.csv'
    df.to_csv(output_csv_file_path, index=False)

    print(f"\nModified CSV file saved to: {output_csv_file_path}")

except FileNotFoundError:
    print(f"File '{csv_file_path}' not found. Please check the file path and try again.")

except KeyError as e:
    print(f"KeyError: {e}. Please check the column name in your CSV file.")

except Exception as e:
    print(f"An error occurred: {e}")

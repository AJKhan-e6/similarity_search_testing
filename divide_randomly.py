import pandas as pd
import random

def divide_csv(file_path, num_rows, run_file='run_data_1000.csv', test_file='test_data_1000.csv'):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Check if num_rows exceeds the number of rows in the file
    if num_rows > len(data):
        raise ValueError(f"The file only contains {len(data)} rows, but {num_rows} rows were requested.")
    
    # Extract the header and shuffle the data
    data_sample = data.sample(n=num_rows, random_state=42).reset_index(drop=True)
    
    # Split the data into two equal parts
    mid_point = len(data_sample) // 2
    run_data = data_sample.iloc[:mid_point]
    test_data = data_sample.iloc[mid_point:]
    
    # Save to new CSV files
    run_data.to_csv(run_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"Data has been divided into '{run_file}' and '{test_file}'.")

# Example usage
file_path = 'classification_nl_output_retry.csv' 
num_rows = 1000  # Specify how many rows to divide
divide_csv(file_path, num_rows)
import pandas as pd
import os

def convert_csv_to_jsonl(input_path, output_path):
    """
    Reads a CSV file and converts it to JSONL format.
    """
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Converting {input_path} to {output_path}...")
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        
        # Write to JSONL (orient='records' with lines=True creates JSONL)
        df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        
        print(f"Successfully converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")

def main():
    # Define input and output paths
    data_dir = 'data'
    
    files_to_convert = [
        ('train.csv', 'train.jsonl'),
        ('test.csv', 'test.jsonl')
    ]

    # Ensure data directory exists
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' does not exist.")
        return

    for input_file, output_file in files_to_convert:
        input_path = os.path.join(data_dir, input_file)
        output_path = os.path.join(data_dir, output_file)
        convert_csv_to_jsonl(input_path, output_path)

if __name__ == "__main__":
    main()
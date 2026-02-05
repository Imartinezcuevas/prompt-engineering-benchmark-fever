import pandas as pd
import os

# Build the path dynamically based on where this script is located
# This ensures it works on Windows, Mac, and Linux
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "src", "data", "raw")

DEFAULT_FILENAME = "train.jsonl"
DEFAULT_PATH = os.path.join(DATA_DIR, DEFAULT_FILENAME)

def load_fever_data(file_path=DEFAULT_PATH, limit=None):
    """
    Loads the FEVER dataset from a local JSON file.
    """
    print(f" Looking for file at: {file_path}")
    
    if not os.path.exists(file_path):
        print(f" Error: File not found.")
        print(f"   I expected to find: '{DEFAULT_FILENAME}' inside '{DATA_DIR}'")
        
        if os.path.exists(DATA_DIR):
            print(f"    The folder exists. Here are the files I can see inside:")
            files = os.listdir(DATA_DIR)
            if not files:
                print("      (The folder is empty)")
            for f in files:
                print(f"      - {f}")
        else:
            print(f"    The folder '{DATA_DIR}' does not exist either.")
            
        return None

    try:
        print("Loading data (this might take a moment)...")
        # Load directly using Pandas (lines=True is crucial for JSONL format)
        df = pd.read_json(file_path, lines=True)
        
        required_columns = ['id', 'claim', 'label', 'evidence']
        
        # Check columns
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Data is missing columns. Found: {df.columns}")
            
        df = df[required_columns]
        df['label'] = df['label'].str.upper()
        
        if limit:
            df = df.head(limit)
            print(f"Dataset limited to {limit} rows.")
            
        print(f"Success! Loaded {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Critical error reading the file: {e}")
        return None

if __name__ == "__main__":
    load_fever_data(limit=5)
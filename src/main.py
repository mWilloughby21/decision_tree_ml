import sys
from pathlib import Path

from helper import get_file_info

FILE = sys.argv[1]
FILE_PATH = Path(f'input/{FILE}')

def main():
    # Parse Data
    n, attributes, labels, training_data = get_file_info(FILE_PATH)
    print("Attributes:", attributes)
    print("Labels:", labels)
    print("Training Data:", training_data)

if __name__ == "__main__":
    main()
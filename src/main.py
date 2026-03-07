import sys
from pathlib import Path

from helper import get_file_info, calc_entropy

FILE = sys.argv[1]
FILE_PATH = Path(f'input/{FILE}')

def main():
    # Parse Data
    n, attributes, labels, training_data = get_file_info(FILE_PATH)
    print("Attributes:", attributes)
    print("Labels:", labels)
    print("Training Data:", training_data)
    
    entropy = calc_entropy(training_data)
    print(entropy)

if __name__ == "__main__":
    main()
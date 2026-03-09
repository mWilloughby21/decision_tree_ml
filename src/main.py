# main.py

import sys
from pathlib import Path

from helper import get_file_info, entropy_from_training, calc_attribute_info, calc_gain, print_tree

FILE = sys.argv[1]
FILE_PATH = Path(f'input/{FILE}')

def main():
    tree_order = {}
    
    # Parse Data
    attributes, labels, training_data = get_file_info(FILE_PATH)
    
    # Calculate entropy
    entropy = entropy_from_training(training_data)
    
    original_attributes = attributes.copy()
    print_attributes = attributes.copy()
    original_attributes_list = list(attributes.keys())
    while attributes:
        # Calculate each attribute info
        attribute_info, attributes = calc_attribute_info(attributes, training_data, labels, original_attributes_list, original_attributes)
        print_attributes.update(attributes)
        # Gain for each attribute
        attribute_gain = calc_gain(entropy, attribute_info)
        
        best_attribute = max(attribute_gain, key=attribute_gain.get)
        best_value = attribute_gain[best_attribute]
        tree_order[best_attribute] = best_value
        attributes.pop(best_attribute)
        
        if len(attributes) == 1:
            key = list(attributes.keys())[0]
            value = attribute_gain[key]
            tree_order[key] = value
            attributes.pop(key)
    
    print_tree(tree_order, print_attributes, training_data)

if __name__ == "__main__":
    main()
# helper.py

import math as m

def get_file_info(path):
    content = path.read_text()
    lines = content.splitlines()

    n = int(lines[0])
    attributes = {}
    labels = []
    training_data = []
    
    for i in range (1, len(lines)):
        row = lines[i].split()
        if i <= n:        # Define attributes
            name = row[0]
            values = list(row[1:])
            attributes[name] = values
        
        elif i == n + 1:  # Define labels
            labels = row[1:]
        
        else:             # Define training data
            training_data.append(row)
    
    return n, attributes, labels, training_data

def calc_labels(training_data):
    result = {}
    
    for record in training_data:
        if record[-1] not in result:
            result[record[-1]] = 1
        else:
            result[record[-1]] += 1
    return result

def calc_entropy(training_data):
    total = 0
    n = len(training_data)
    label_count = calc_labels(training_data)
    
    for amount in label_count.values():
        p = amount / n
        total -= p * m.log2(p)
    return total
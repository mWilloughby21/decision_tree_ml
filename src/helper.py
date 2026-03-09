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
    
    return attributes, labels, training_data

def calc_labels(training_data):
    result = {}
    
    for record in training_data:
        if record[-1] not in result:
            result[record[-1]] = 1
        else:
            result[record[-1]] += 1
    return result

def calc_entropy(values):
    total = 0
    
    for value in values:
        p = value / sum(values)
        if p > 0:
            total -= p * m.log2(p)
    
    return total

def entropy_from_training(training_data):
    total = 0
    n = len(training_data)
    label_amounts = calc_labels(training_data)
    
    for amount in label_amounts.values():
        p = amount / n
        total -= p * m.log2(p)
    return total

def entropy_from_dicts(attribute_counts):
    result = {}
    
    for i, attribute in enumerate(attribute_counts.values()):
        total = 0
        n = 0
        
        for values in attribute.values():
            for value in values:
                n += value
        for values in attribute.values():
            total += (sum(values) / n) * calc_entropy(values)
        result[list(attribute_counts.keys())[i]] = total
    return result

def create_attribute_info_dict(attributes, training_data, labels):
    result = {}
    
    for i, (key, attribute) in enumerate(attributes.items()):
        value_counts = {}
        
        for value in attribute:
            value_counts[value] = [0] * len(labels)
            
        for record in training_data:
            value = record[i]
            ans = int(record[-1])
            value_counts[value][ans] += 1
        
        result[key] = value_counts
    return result

def calc_attribute_info(attributes, training_data, labels):
    attribute_counts = create_attribute_info_dict(attributes, training_data, labels)
    entropy = entropy_from_dicts(attribute_counts)
    return entropy

def calc_gain(entropy, attribute_info):
    result = {}
    for (key, value) in attribute_info.items():
        result[key] = entropy - value
    return result

def majority_label(training_data):
    counts = {}
    
    for record in training_data:
        label = record[-1]
        counts[label] = counts.get(label, 0) + 1
    
    return max(counts, key=counts.get)

def print_tree(tree_order, attributes, training_data, indent=0):
    keys = list(tree_order.keys())
    
    if not keys:
        return
    
    attribute = keys[0]
    attribute_index = list(attributes.keys()).index(attribute)
    
    remaining = dict(list(tree_order.items())[1:])
    
    for value in attributes[attribute]:
        subset = [row for row in training_data if row[attribute_index] == value]
        
        if not subset:
            label = majority_label(training_data)
            print(" " * indent + f"{attribute}={value}: ({tree_order[attribute]})")
            print(" " * (indent + 2) + label)
            continue
        
        print(" " * indent + f"{attribute}={value}: ({tree_order[attribute]})")
        
        if remaining:
            print_tree(remaining, attributes, subset, indent + 2)
        else:
            label = majority_label(subset)
            print(" " * (indent + 2) + label)
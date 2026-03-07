# helper.py

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
import json
import os

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)  # indent for pretty print

# List of json filenames you want to merge
data_dir = "D:\\FEI-STU\\TP\\network\\client\\dataset"
json_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]

merged_data = []

for filename in json_filenames:
    data = load_json(filename)
    merged_data.extend(data)  # extend because each file contains a list of image data

# Save merged data to a new JSON file
save_json(merged_data, 'merged.json')

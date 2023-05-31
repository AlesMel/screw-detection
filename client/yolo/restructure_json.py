import json
import os
import re

# Function to get image_id from filename
def get_image_id(filename):
    match = re.search(r'screws_(\d+).json', filename)
    if match:
        return int(match.group(1))
    return None

# directory of your json files
dir_path = "client/dataset"

# get a list of all json files in the directory
json_filenames = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.json')]

categories = []
images = []
annotations = []
category_ids = {}
annotation_id = 1001

for filename in json_filenames:
    # Load original JSON
    with open(filename) as f:
        data = json.load(f)
    
    image_id = get_image_id(filename)  # Extract image_id from filename
    
    # Assuming each file has data for one image
    image_data = data[0]  # Access the first (and only) dictionary in the list
    
    # Add image info
    images.append({
        "file_name": image_data["image"],
        "height": 640,  # Placeholder, replace with actual height
        "width": 640,  # Placeholder, replace with actual width
        "id": image_id,
        "license": 1  # Placeholder, replace with actual license if needed
    })

    for annotation in image_data['annotations']:
        # Create category if it doesn't exist yet
        label = annotation['label']
        if label not in category_ids:
            category_id = len(categories) + 1
            category_ids[label] = category_id
            categories.append({
                "id": category_id,
                "name": label,
                "supercategory": "object"
            })

        # Create annotation
        coordinates = annotation['coordinates']
        bbox = [coordinates['x'], coordinates['y'], coordinates['width'], coordinates['height'], 0]
        area = coordinates['width'] * coordinates['height']
        annotations.append({
            "area": area,
            "bbox": bbox,
            "category_id": category_ids[label],
            "id": annotation_id,
            "image_id": image_id,  # Assign image_id based on filename
            "is_crowd": 0
        })
        annotation_id += 1

# Create new JSON structure
new_data = {
    "categories": categories,
    "images": images,
    "annotations": annotations
}

# Save new JSON
with open('new_screws.json', 'w') as f:
    json.dump(new_data, f, indent=4)

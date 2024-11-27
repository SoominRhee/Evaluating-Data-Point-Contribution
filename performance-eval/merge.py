import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
influence_scores_path = os.path.join(current_dir, 'TracIn_influence_scores.json')
target_points_info_path = os.path.join(current_dir, 'TracIn_target_points_info.json')
output_path = os.path.join(current_dir, 'merged_output.json')

with open(influence_scores_path, 'r') as f:
    influence_scores = json.load(f)

with open(target_points_info_path, 'r') as f:
    target_points_info = json.load(f)

results = []

for item in target_points_info:
    index = str(item['index'])
    if index in influence_scores:
        result_item = {
            "image": item["image"],
            "overall_influence": influence_scores[index]["overall_influence"]
        }
        results.append(result_item)

with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Merged data saved to {output_path}")

import json

with open('brisque_scores.json', 'r') as f:
    data = json.load(f)

sorted_data = sorted(data, key=lambda x: x['score'], reverse=False)

with open('brisque_scores_sorted.json', 'w') as f:
    json.dump(sorted_data, f, indent=4)


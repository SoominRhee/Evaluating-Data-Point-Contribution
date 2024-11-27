import json
import pandas as pd

with open('individual_influence_sorted.json', 'r') as f:
    individual_influence_data = json.load(f)

with open('clip_similarity_sorted1.json', 'r') as f:
    clip_similarity_data = json.load(f)

individual_influence_images = [entry['image'] for entry in individual_influence_data]
clip_similarity_images = [entry['image'] for entry in clip_similarity_data]

df_comparison = pd.DataFrame({
    'Individual Influence Rank': range(1, len(individual_influence_images) + 1),
    'Image': individual_influence_images,
    'CLIP Similarity Rank': [clip_similarity_images.index(img) + 1 for img in individual_influence_images]
})

df_comparison['Rank Difference'] = df_comparison['Individual Influence Rank'] - df_comparison['CLIP Similarity Rank']

print(df_comparison)

df_comparison.to_csv('rank_comparison1.csv', index=False)

import json
import matplotlib.pyplot as plt
import numpy as np

with open('clip_overall_similarity_sorted.json', 'r') as f:
    clip_data = json.load(f)

with open('influence_scores_sorted.json', 'r') as f:
    influence_data = json.load(f)

clip_ranking = {item['image']: rank for rank, item in enumerate(clip_data, start=1)}
influence_ranking = {item['image']: rank for rank, item in enumerate(influence_data, start=1)}

images_sorted_by_clip = [item['image'] for item in clip_data]

clip_ranks = [clip_ranking[img] for img in images_sorted_by_clip]
influence_ranks = [influence_ranking[img] for img in images_sorted_by_clip]

plt.figure(figsize=(12, 6))
plt.plot(clip_ranks, label='CLIP Similarity Rank', marker='o', color='blue')
plt.plot(influence_ranks, label='Influence Score Rank', marker='x', color='orange')
plt.xticks(ticks=np.arange(len(images_sorted_by_clip)), labels=images_sorted_by_clip, rotation=90)
plt.xlabel('Image (Sorted by CLIP Rank)')
plt.ylabel('Rank')
plt.title('Rank Comparison between CLIP Similarity and Influence Score')
plt.legend()
plt.tight_layout()

plt.savefig('clip_ranking_comparison.png')

ranking_comparison = [{'image': img, 'clip_rank': clip_ranking[img], 'influence_rank': influence_ranking[img]} for img in images_sorted_by_clip]
with open('ranking_comparison.json', 'w') as f:
    json.dump(ranking_comparison, f, indent=4)

plt.show()

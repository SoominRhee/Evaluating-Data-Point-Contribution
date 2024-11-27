import json
import matplotlib.pyplot as plt
import numpy as np

with open('brisque_scores_sorted.json', 'r') as f:
    brisque_data = json.load(f)

with open('influence_scores_sorted.json', 'r') as f:
    influence_data = json.load(f)

brisque_ranking = {item['image']: rank for rank, item in enumerate(brisque_data, start=1)}
influence_ranking = {item['image']: rank for rank, item in enumerate(influence_data, start=1)}

images_sorted_by_brisque = [item['image'] for item in brisque_data]

brisque_ranks = [brisque_ranking[img] for img in images_sorted_by_brisque]
influence_ranks = [influence_ranking[img] for img in images_sorted_by_brisque]

plt.figure(figsize=(12, 6))
plt.plot(brisque_ranks, label='BRISQUE Rank', marker='o', color='blue')
plt.plot(influence_ranks, label='Influence Rank', marker='x', color='orange')

plt.xticks(ticks=np.arange(len(images_sorted_by_brisque)), labels=images_sorted_by_brisque, rotation=90)

plt.xlabel('Image (Sorted by BRISQUE Rank)')
plt.ylabel('Rank')
plt.title('Rank Comparison between Brisque Score and Influence Score')
plt.legend()

plt.tight_layout()
plt.savefig('brisque_vs_influence_ranking_comparison.png')
plt.show()

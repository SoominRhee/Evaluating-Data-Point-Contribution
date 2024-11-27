import json
import matplotlib.pyplot as plt

with open('3k_img1_dataset_compare.json', 'r') as f:
    dataset_compare = json.load(f)

with open('3k_img1_image_similarity.json', 'r') as f:
    image_similarity = json.load(f)

with open('3k_img1_text_similarity.json', 'r') as f:
    text_similarity = json.load(f)

dataset_compare_sorted = sorted(dataset_compare, key=lambda x: x['lpips_score'], reverse=True)
sorted_datasets = [item['Dataset'] for item in dataset_compare_sorted]
dataset_compare_ranks = {item['Dataset']: rank for rank, item in enumerate(dataset_compare_sorted, start=1)}

image_similarity_sorted = sorted(image_similarity, key=lambda x: x['average_image_similarity'])
image_similarity_ranks = {item['dataset']: rank for rank, item in enumerate(image_similarity_sorted, start=1)}

text_similarity_sorted = sorted(text_similarity, key=lambda x: x['average_text_similarity'], reverse=True)
text_similarity_ranks = {item['dataset']: rank for rank, item in enumerate(text_similarity_sorted, start=1)}

lpips_ranks = list(range(1, 6)) 
image_sim_ranks = [image_similarity_ranks[ds] for ds in sorted_datasets]
text_sim_ranks = [text_similarity_ranks[ds] for ds in sorted_datasets]

plt.figure(figsize=(10, 5))
plt.plot(sorted_datasets, lpips_ranks, label='Dataset Contribution Rank', marker='o', color='blue')
plt.plot(sorted_datasets, image_sim_ranks, label='Image Similarity Rank', marker='x', color='orange')
plt.xlabel('Dataset')
plt.ylabel('Rank')
plt.title('Rank Comparison between Contribution and Image Similarity')
plt.yticks(range(1, 11))  
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('3k_dataset_image_rank_img1.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(sorted_datasets, lpips_ranks, label='Dataset Contribution Rank', marker='o', color='blue')
plt.plot(sorted_datasets, text_sim_ranks, label='Text Similarity Rank', marker='x', color='green')
plt.xlabel('Dataset')
plt.ylabel('Rank')
plt.title('Rank Comparison between Contribution and Text Similarity')
plt.yticks(range(1, 11))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('3k_dataset_text_rank_img1.png')
plt.show()

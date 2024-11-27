import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from dataset import TextImageDataset
from model import TextToImageModel
import os
import random
import json
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TracInWD:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def get_loss_gradient(self, inputs, targets):
        self.model.eval()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        grads = [param.grad.clone().detach() for param in self.model.parameters() if param.grad is not None]
        return grads

    def calculate_influence(self, target_input, target_output, dataloader):
        influences = []
        target_grads = self.get_loss_gradient(target_input, target_output)

        for batch in dataloader:
            input_ids, pixel_values = batch
            input_ids, pixel_values = input_ids.to(self.device), pixel_values.to(self.device)
            grads = self.get_loss_gradient(input_ids, pixel_values)
            influence = sum([(tg * g).sum().item() for tg, g in zip(target_grads, grads)])
            influences.append(influence)

        return influences

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'dataset')
image_folder = os.path.join(dataset_path, 'images')
captions_file = os.path.join(dataset_path, 'captions.txt')

dataset = TextImageDataset(image_folder=image_folder, captions_file=captions_file, processor=processor, num_images=1000)

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    pixel_values = [item[1] for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    pixel_values_padded = torch.stack(pixel_values)
    return input_ids_padded, pixel_values_padded

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

model = TextToImageModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("text2image_model5.pth", map_location=device))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
tracin = TracInWD(model, criterion, optimizer, device)

random.seed(42)  
unique_image_indices = random.sample(range(len(dataset) // 5), 50)  

target_points = []
target_info = []
for idx in unique_image_indices:
    img_idx = idx * 5
    img_info = {"image": dataset.captions.iloc[img_idx]['file_name'], "captions": [], "index": img_idx}
    target_data = []
    for i in range(5):
        data_idx = img_idx + i
        target_data.append((dataset[data_idx][0].unsqueeze(0).to(device), dataset[data_idx][1].unsqueeze(0).to(device), data_idx))
        img_info["captions"].append(dataset.captions.iloc[data_idx]['caption'])
    target_points.append(target_data)
    target_info.append(img_info)

with open("target_points_info.json", "w") as f:
    json.dump(target_info, f, indent=4)

influence_scores = []
for target_data in target_points:
    influences_per_caption = []
    
    for input_ids, pixel_values, idx in target_data:
        influences = tracin.calculate_influence(input_ids, pixel_values, dataloader)
        influences_per_caption.append(np.mean(influences))  
        print(f"Data point {idx} influence score: {np.mean(influences):.4f}")
        
    overall_influence = np.mean(influences_per_caption)
    influence_scores.append({
        "image": dataset.captions.iloc[target_data[0][2] // 5 * 5]['file_name'],
        "overall_influence": overall_influence
    })
    print(f"Image {dataset.captions.iloc[target_data[0][2] // 5 * 5]['file_name']} overall influence score: {overall_influence:.4f}")

with open("influence_scores.json", "w") as f:
    json.dump(influence_scores, f, indent=4)

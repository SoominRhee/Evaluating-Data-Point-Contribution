import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
import torch.optim as optim
from dataset import TextImageDataset
from model import TextToImageModel
import os
import matplotlib.pyplot as plt
import csv
import time

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'dataset')
image_folder = os.path.join(dataset_path, 'images')
captions_file = os.path.join(dataset_path, 'captions.txt')
dataset = TextImageDataset(image_folder=image_folder, captions_file=captions_file, processor=processor, num_images=3000)

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

print(f"Using device: {device}")

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

start_time = time.time()

losses = []

csv_file = "training_losses.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss"])  

    for epoch in range(300):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            input_ids, pixel_values = batch
            input_ids, pixel_values = input_ids.to(device), pixel_values.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, pixel_values)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        writer.writerow([epoch + 1, avg_loss])

torch.save(model.state_dict(), "text2image_3k.pth")

end_time = time.time()
training_time = end_time - start_time
print(f"Total Training Time: {training_time:.2f} seconds")

plt.plot(range(1, 301), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.savefig('loss_over_epochs.png')

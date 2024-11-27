import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
import torch.optim as optim
from dataset import TextImageDataset
from model import TextToImageModel
import os
import matplotlib.pyplot as plt
import json
import csv
import time

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'dataset')
image_folder = os.path.join(dataset_path, 'images')
captions_file = os.path.join(dataset_path, 'captions.txt')

original_dataset = TextImageDataset(image_folder=image_folder, captions_file=captions_file, processor=processor, num_images=3000)

def filter_dataset(dataset, images_to_remove):
    filtered_samples = []
    removed_info = []
    for i in range(len(dataset)):
        image_file = dataset.captions.iloc[i]['file_name']  
        if image_file not in images_to_remove:
            filtered_samples.append(dataset[i])
        else:
            caption = dataset.captions.iloc[i]['caption'] 
            removed_info.append({
                "image": image_file,
                "captions": caption
            })
    return filtered_samples, removed_info

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    pixel_values = [item[1] for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    pixel_values_padded = torch.stack(pixel_values)
    return input_ids_padded, pixel_values_padded

def train_and_save_model(start_index, end_index, target_json, csv_file, model_path, graph_path):
    image_names_to_remove = sorted(os.listdir(image_folder))[start_index:end_index]

    filtered_dataset, removed_info = filter_dataset(original_dataset, image_names_to_remove)

    with open(target_json, "w") as f:
        json.dump(removed_info, f, indent=4)
        f.write("\nRemoved images:\n")
        for img_name in image_names_to_remove:
            f.write(f'"{img_name}", ')

    filtered_dataloader = DataLoader(filtered_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = TextToImageModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    start_time = time.time()

    losses = []

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"]) 

        for epoch in range(300):
            model.train()
            running_loss = 0.0
            for batch in filtered_dataloader:
                input_ids, pixel_values = batch
                input_ids, pixel_values = input_ids.to(device), pixel_values.to(device)
                optimizer.zero_grad()
                outputs = model(input_ids)
                loss = criterion(outputs, pixel_values)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(filtered_dataloader)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

            writer.writerow([epoch + 1, avg_loss])

    torch.save(model.state_dict(), model_path)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total Training Time: {training_time:.2f} seconds")

    plt.plot(range(1, 301), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.savefig(graph_path)

ranges = [
    (100, 200, "3k_f2_target.json", "training_losses_f2.csv", "3k_f2.pth", "loss_over_epochs_f2.png"),
    (200, 300, "3k_f3_target.json", "training_losses_f3.csv", "3k_f3.pth", "loss_over_epochs_f3.png"),
    (300, 400, "3k_f4_target.json", "training_losses_f4.csv", "3k_f4.pth", "loss_over_epochs_f4.png"),
    (400, 500, "3k_f5_target.json", "training_losses_f5.csv", "3k_f5.pth", "loss_over_epochs_f5.png")
]

for start_index, end_index, target_json, csv_file, model_path, graph_path in ranges:
    train_and_save_model(start_index, end_index, target_json, csv_file, model_path, graph_path)

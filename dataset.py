import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from transformers import CLIPProcessor

class TextImageDataset(Dataset):
    def __init__(self, image_folder, captions_file, processor, num_images=1000):
        self.image_folder = image_folder
        self.processor = processor
        
        captions = pd.read_csv(captions_file, header=None, names=['file_name', 'caption'])
        
        unique_images = captions['file_name'].unique()[:num_images]
        self.captions = captions[captions['file_name'].isin(unique_images)]

        print(f"Loaded {len(self.captions)} captions")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.captions.iloc[idx, 0].strip())

        if os.path.isfile(img_name):
            image = Image.open(img_name).convert("RGB")
            caption = self.captions.iloc[idx, 1].strip()
            inputs = self.processor(text=[caption], images=image, return_tensors="pt", padding=True)
            return inputs.input_ids.squeeze(), inputs.pixel_values.squeeze()
        else:
            print(f"File not found: {img_name}, skipping this file.")
            idx = (idx + 1) % len(self.captions)
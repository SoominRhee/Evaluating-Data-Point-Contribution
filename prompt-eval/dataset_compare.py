import torch
import lpips
from PIL import Image
import os
import json
from torchvision import transforms

lpips_loss_fn = lpips.LPIPS(net='alex')
lpips_loss_fn.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_image(image_name):
    image_path = os.path.join(current_dir, image_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

reference_image_name = "3k_img1.png"  
reference_image = load_image(reference_image_name)

image_names = [
    "3k_img1_f1.png",
    "3k_img1_f2.png",
    "3k_img1_f3.png",
    "3k_img1_f4.png",
    "3k_img1_f5.png",
    ]

results = []

for idx, image_name in enumerate(image_names, 1):
    target_image = load_image(image_name)
    similarity = lpips_loss_fn(reference_image, target_image).item()
    results.append({
        "Dataset": f"Dataset {idx}",
        "lpips_score": similarity
    })

output_file = "3k_img1_dataset_compare.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Analysis complete. Results saved to {output_file}.")

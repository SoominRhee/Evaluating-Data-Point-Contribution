import torch
import matplotlib.pyplot as plt
from transformers import CLIPProcessor
from model import TextToImageModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "text2image_3k.pth")

model = TextToImageModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_image(text, output_path):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        output_image = model(input_ids)
    image = output_image.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

generate_image("A dog catches a red ball in mid-air near a lake.", "3k_img1_test.png")

#input_text = "A boy in a green shirt diving into a lake." 
#input_text = "A boy wearing a green shirt jumps to catch a Frisbee."
#input_text = "A dog catches a red ball in mid-air near a lake." #img1
#input_text = "A person wearing a blue helmet climbs a rocky mountain." 
#input_text = "a girl eats a food in front of trees." #img2

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class FullyConnectedDecoder(nn.Module):
    def __init__(self, text_embedding_dim, image_size=224):
        super(FullyConnectedDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 3 * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, text_embedding):
        x = self.fc(text_embedding)
        x = x.view(-1, 3, 224, 224)
        return x

class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.decoder = FullyConnectedDecoder(self.clip_model.config.projection_dim)

    def forward(self, input_ids):
        text_features = self.clip_model.get_text_features(input_ids)
        image_features = self.decoder(text_features)
        return image_features

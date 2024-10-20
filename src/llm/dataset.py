import base64
import numpy as np
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, notes, prompt_template:str, text_to_replace:str = "{section}"):
        self.notes = notes
        self.prompt_template = prompt_template
        self.text_to_replace = text_to_replace

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        note = self.notes[idx]
        prompt = self.prompt_template.replace(self.text_to_replace, note)
        return prompt

class ImageDataset(Dataset):
    def __init__(self, image_paths, prompt_template):
        self.image_paths = image_paths
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.image_paths)

    # Note: this is a payload for OpenAI API models
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        payload = [
            {
                "type": "text",
                "text": self.prompt_template
            }
        ]
        return payload, image_path


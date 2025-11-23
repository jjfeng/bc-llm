import base64
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from langchain_core.messages import HumanMessage, SystemMessage

class TextDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

class ImageDataset(Dataset):
    def __init__(self, image_paths, prompt_template, prompts = None):
        self.image_paths = image_paths
        if prompts is not None:
            self.prompts = prompts
        else:
            self.prompts = [prompt_template] * len(image_paths)

    def __len__(self):
        return len(self.image_paths)

    # Note: this is a payload for OpenAI API models
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # with open(image_path, "rb") as image_file:
        #     base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        payload = [
            {
                "type": "text",
                "text": self.prompts[idx]
            }
            # ,{
            #     "type": "image_url",
            #     "image_url": {
            #         "url": f"data:image/jpeg;base64,{base64_image}"
            #     }
            # }
        ]
        return payload, image_path

class ImageGroupDataset(Dataset):
    def __init__(self, image_paths_list, prompt_template=None, prompts = None):
        self.image_paths_list = image_paths_list
        if prompts is not None:
            self.prompts = prompts
        else:
            self.prompts = [prompt_template] * len(image_paths_list)

    def __len__(self):
        return len(self.image_paths_list)

    # Note: this is a payload for OpenAI API models
    def __getitem__(self, idx):
        image_paths = self.image_paths_list[idx]
        payload = [
            {
                "type": "text",
                "text": self.prompts[idx]
            }
        ]
        # for image_path in image_paths:
        #     with open(image_path, "rb") as image_file:
        #         base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        #     payload.append({
        #         "type": "image_url",
        #         "image_url": {
        #             "url": f"data:image/jpeg;base64,{base64_image}",
        #             "detail": "low"
        #         }
        #     })
        return payload, "+".join(image_paths)


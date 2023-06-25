import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import random

class LightDataset(Dataset):
    def __init__(self, data_path, transform, tokenizer, test=False):
        self.data_path = data_path
        self.class_to_id = {"green": "a traffic light with lighted green bulb",
                            "red": "a traffic light with lighted red bulb",
                            "black": "a traffic light with  no lighted bulbs",
                            "yellow": "a traffic light with lighted yellow bulb",
                            "number": "a traffic light with lighted number bulb"}
        self.image_paths = []
        self.labels = []
        self.crop_size = (96, 96)
        self.test = test
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Loop through each folder and gather image paths and labels
        for folder in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, folder)) and folder in self.class_to_id.keys():
                class_text = self.class_to_id[folder]
                folder_path = os.path.join(data_path, folder)
                for image_name in os.listdir(folder_path):
                    if image_name.endswith(".jpg"):
                        image_path = os.path.join(folder_path, image_name)
                        self.image_paths.append(image_path)
                        self.labels.append(class_text)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        example = {}
        
        label = self.labels[index]
        example["input_ids"] = self.tokenizer(
            label,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            ).input_ids[0]
        
        image_path = self.image_paths[index]
        img = Image.open(image_path)
        img = img.resize(self.crop_size, resample=Image.BILINEAR)
        img = self.transform(img)
        example["pixel_values"] = img
        return example


if __name__ == '__main__':
    # 定义数据集路径
    data_path = "/data_4/jyj/datasets/SinLight"

    # # 创建数据集
    dataset = LightDataset(data_path)
    
    from torch.utils.data import DataLoader

    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(len(dataloader))
    
    for batch in dataloader:
        images, labels = batch
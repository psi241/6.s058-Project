#!/usr/bin/env python
# coding: utf-8
# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision.models as models
from torchvision import transforms
import numpy as np
import random
from tqdm import tqdm
from utils.dataloader import *
import random
import matplotlib.pyplot as plt
import pprint
from model.models import SimCLRModel

# In[4]:

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()  
DATASET_PATH = os.getenv('DATASET_PATH')
manga_name_list = get_book_list(DATASET_PATH)

np.random.seed(12)

to_tensor = transforms.ToTensor()


class Manga109Dataset(Dataset):
    def __init__(self, dataset_path, manga_name, transform=jitter_box):
        self.dataset_path = dataset_path
        self.manga_name = manga_name
        self.transform = transform

        annotations = annotation_loader(dataset_path, manga_name)
        self.page_width = int(annotations['book']['pages']['page'][0]['@width'])
        self.page_height = int(annotations['book']['pages']['page'][0]['@height'])

        self.char_ids = []
        self.char_faces = {}

        for char_id in get_character_list(annotations)[0]:
            faces_info, _ = get_crops_info_char(annotations, char_id)
            if len(faces_info) >= 2:
                self.char_ids.append(char_id)
                self.char_faces[char_id] = faces_info

    def __len__(self):
        return len(self.char_ids)

    def __getitem__(self, index):
        char_id = self.char_ids[index]
        faces_info = self.char_faces[char_id]

        crop_i, crop_j = random.sample(faces_info, 2)

        if self.transform:
            pos_i = jitter_box(crop_i['position'], self.page_width, self.page_height)
            pos_j = jitter_box(crop_j['position'], self.page_width, self.page_height)
        else:
            pos_i = crop_i['position']
            pos_j = crop_j['position']

        xi = to_tensor(retrieve_page(self.dataset_path, self.manga_name, crop_i['@index'], pos_i, target_size=(112, 112)))
        xj = to_tensor(retrieve_page(self.dataset_path, self.manga_name, crop_j['@index'], pos_j, target_size=(112, 112)))
        return xi, xj

def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T)
    N = z_i.shape[0]

    mask = (~torch.eye(2*N, dtype=bool)).to(z.device)
    sim = similarity / temperature
    exp_sim = torch.exp(sim) * mask

    positive_sim = torch.exp(F.cosine_similarity(z_i, z_j) / temperature)
    positives = torch.cat([positive_sim, positive_sim], dim=0)

    denominator = exp_sim.sum(dim=1)
    loss = -torch.log(positives / denominator)
    return loss.mean()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)

# In[15]:


model = SimCLRModel()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

trained_manga_indices = list(range(16))
train_loaders = {}

for manga_idx in trained_manga_indices:
    manga_name = manga_name_list[manga_idx]

    print(manga_name)
    contrastive_dataset = Manga109Dataset(DATASET_PATH, manga_name)
    train_loaders[manga_idx] = DataLoader(contrastive_dataset, batch_size=1, shuffle=True, num_workers=0)

model.train()
total_loss = 0

for epoch in range(10):
    epoch_loss = 0
    counter = 0
    for manga_idx in tqdm(trained_manga_indices):
        train_loader = train_loaders[manga_idx]
        
        for x_i, x_j in train_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i = model(x_i)
            z_j = model(x_j)
            optimizer.zero_grad()
            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            counter += len(train_loader)

    print(f"Epoch {epoch+1} | Loss: {epoch_loss / counter:.4f}")

model = model.cpu()
torch.save(model.state_dict(), "simclr.pt")


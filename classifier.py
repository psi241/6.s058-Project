import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision.models as models
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils.dataloader import *

from dotenv import load_dotenv
import os
from model.models import SimCLRModel
from params import DATASET_PATH, manga_name_list, TARGET_SIZE

class CharacterFaceLabelLoader(Dataset):
    def __init__(self, imgs, labels, transform):
        '''
        Arguments
            X: list of PIL images in RGB
            Y: 1-D list/numpy array of Label
        '''
        # annotations = annotation_loader(dataset_path, manga_name)
        # chars_dict = create_data(dataset_path, annotations, target_size = TARGET_SIZE)
        # X = []
        # Y = []
        # self.map_idx_to_id = {}
        # for i, (char_id, char_dict) in enumerate(chars_dict.items()):
        #     self.map_idx_to_id[i] = char_id
        #     for img in char_dict['face_imgs']:
        #         transformed_img = transform(img)
        #         Y.append(torch.tensor(i))
        #         X.append(transformed_img)

        X = [transform(img) for img in imgs]
        self.imgs = torch.stack(X, dim=0)
        self.labels = torch.tensor(labels)

        self.encoded_imgs = None

    def __len__(self):
        return self.imgs.shape[0]
    
    def set_encoded_imags(self, model, batch_size = 32):
        '''
        Compute encoding of images in batches using PyTorch
        
        Args:
            model: The encoding model (PyTorch model)
            batch_size (int): Batch size for processing
        
        Returns:
            int: Number of images encoded
        '''
        
        if len(self.imgs) == 0:
            self.encoded_imgs = torch.tensor([])
            return 0
        
        encoded_results = []
        num_images = len(self.imgs)
        model.eval()  # Set model to evaluation mode

        pbar = tqdm(range(0, num_images, batch_size), desc = "batch")
        
        with torch.no_grad():  # No gradient computation
            for i in pbar:
                batch_end = min(i + batch_size, num_images)
                batch = self.imgs[i:batch_end]
                
                # Encode batch
                encoded_batch = model(batch)
                encoded_results.append(encoded_batch)
        
        # Concatenate all batches
        self.encoded_imgs = torch.cat(encoded_results, dim=0)
        
        return num_images

    def __getitem__(self, index):
        if self.encoded_imgs is None:
            raise ValueError("Image needed to be encoded first")
        return self.encoded_imgs[index], self.labels[index]
    
# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
# Retrieved on 5 May 2026
def train_one_epoch(epoch_index, model, training_loader, optimizer):
    running_loss = 0.
    last_loss = 0.
    
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.
    count = 0
    for i, data in enumerate(training_loader):
        inputs, labels = data
        count += len(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        total_loss += loss.detach().cpu()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, running_loss / 1000))
            running_loss = 0.
    print("Epoch {}, avg loss = {}".format(epoch_index, total_loss/count))

    return last_loss

def make_prediction(manga_idx, train_samples, train_labels):
    pass


if __name__ == "__main__":
    print("[prog] Model Load")
    model = SimCLRModel()
    model.load_state_dict(torch.load('simclr.pt', weights_only=True))
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("[prog] Data Load")
    annotations = annotation_loader(DATASET_PATH, 'AisazuNihaIrarenai')
    chars_dict = create_data(DATASET_PATH, annotations, target_size = TARGET_SIZE)
    X = []
    Y = []
    map_idx_to_id = {}
    for i, (char_id, char_dict) in enumerate(chars_dict.items()):
        map_idx_to_id[i] = char_id
        for img in char_dict['face_imgs']:
            Y.append(i)
            X.append(img)

    dataloader = CharacterFaceLabelLoader(DATASET_PATH, X, Y, simple_transform)

    print("[prog] Encoding")
    dataloader.set_encoded_imags(model)

    classifier = nn.Sequential(
        nn.Linear(model.projection_dim , 256),
        nn.Linear(256 , dataloader.labels_count),
    )

    training_set, test_set = torch.utils.data.random_split(dataloader, [0.7, 0.3])

    print("[prog] Train classifier")
    NUM_EPOCH = 50
    train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0)
    for i in range(NUM_EPOCH):
        train_one_epoch(i, classifier, train_loader, optimizer)

    correct_count = 0
    counter = 0

    print("[prog] Evaluation")
    for encoded, label in test_set:
        logits = classifier(encoded)
        prob = torch.softmax(logits, dim = 0)
        prediction = torch.argmax(prob, dim = 0).item()
        print(prediction, label)
        counter += 1
        if prediction == label.item():
            correct_count += 1
    
    print("Total accuracy = ", correct_count/counter)

    # print(len(dataloader))
    # x, y = dataloader[17]
    # print("shape ", x.shape)
    # print("label ", y)

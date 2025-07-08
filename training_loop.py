import numpy as np
import os
from pathlib import Path
from PIL import Image
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms as T
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

from CellDataset import CellDataset
from MoCoResNetBackbone import MoCoResNetBackbone
from MoCoV2Loss import MoCoV2Loss


if __name__ == '__main__':
    modelPath = Path("/scratch/cv-course-group-5/models/")

    trainingName = sys.argv[1] if len(sys.argv) >= 2 else "training/"

    gpu = sys.argv[2] if len(sys.argv) >= 3 else 0

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = MoCoResNetBackbone()
    model.to(device)
    
    dataset = CellDataset()

    moco_loss = MoCoV2Loss(device=device)

    epochs = 50
    batch_size = 64
    learning_rate = 0.001
    momentum = 0.9

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,        # Adjust to CPU core count
        pin_memory=True,       # Enables fast transfer to GPU
    )

    losses = [[] for _ in range(epochs)]

    checkpoint_epoch = 0
    while os.path.exists(modelPath / trainingName / f"model_epoch{checkpoint_epoch + 5}.pth"):
        checkpoint_epoch = checkpoint_epoch + 5

    if checkpoint_epoch > 0:
        model_state_dict = torch.load(modelPath / trainingName / f"model_epoch{checkpoint_epoch}.pth")
        model.load_state_dict(model_state_dict)
        loss_state_dict = torch.load(modelPath / trainingName / f"model_epoch{checkpoint_epoch}.pth")

    for epoch in range(checkpoint_epoch + 1, epochs):

        model.train()
        for [keys, queries] in tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader), ncols=100):

            keys = keys.to(device, non_blocking=True)
            queries = queries.to(device, non_blocking=True)

            query_encodings, key_encodings = model(queries, keys)

            loss = moco_loss(query_encodings, key_encodings)
            losses[epoch].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            if not os.path.exists(modelPath / trainingName):
                os.makedirs(modelPath / trainingName)
            torch.save(model.state_dict(), modelPath / trainingName / f"model_epoch{epoch}.pth")
            torch.save(moco_loss.state_dict(), modelPath / trainingName / f"loss_epoch{epoch}.pth")

        print(f"Epoch {epoch} loss: {sum(losses[epoch]) / len(losses[epoch])} ")
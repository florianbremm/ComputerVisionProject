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

from CellDataset import CellDataset, moco_transform
from MoCoResNetBackbone import MoCoResNetBackbone
from MoCoV2Loss import MoCoV2Loss


if __name__ == '__main__':
    modelPath = Path("/scratch/cv-course-group-5/models/")

    # training Name from parameters
    trainingName = sys.argv[1] if len(sys.argv) >= 2 else "training/"

    # initialize device from parameters
    gpu = sys.argv[2] if len(sys.argv) >= 3 else 0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # load video list specified in the parameters
    video_path = Path(sys.argv[3] if len(sys.argv) >= 4 else 'reduced_videos.json')
    with open(video_path, 'r') as f:
        video_list = json.load(f)

    # training parameters
    epochs = 50
    batch_size = 128
    learning_rate = 0.005
    momentum = 0.9

    #init model dataset, loss, optimizer, and dataloader
    model = MoCoResNetBackbone()
    model.to(device)

    dataset = CellDataset(video_list=video_list, transform=moco_transform)

    moco_loss = MoCoV2Loss(device=device, queue_size=8129)

    optimizer = torch.optim.SGD(
        model.encoder_q.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=1e-4,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,        # Adjust to CPU core count
        pin_memory=True,       # Enables fast transfer to GPU
    )

    losses = [[] for _ in range(epochs)]

    # check for checkpoint of the model (in case the training was interrupted we don't have to start from scratch
    checkpoint_epoch = 0
    while os.path.exists(modelPath / trainingName / f"model_epoch{checkpoint_epoch + 5}.pth"):
        checkpoint_epoch = checkpoint_epoch + 5

    # load model and loss state from checkpoint
    if checkpoint_epoch > 0:
        model_state_dict = torch.load(modelPath / trainingName / f"model_epoch{checkpoint_epoch}.pth")
        model.load_state_dict(model_state_dict)
        loss_state_dict = torch.load(modelPath / trainingName / f"loss_epoch{checkpoint_epoch}.pth")
        moco_loss.load_state_dict(loss_state_dict)

    for epoch in range(checkpoint_epoch + 1, epochs + 1):

        model.train()
        for [keys, queries] in tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader), ncols=100):

            #encode keys and queries
            keys = keys.to(device, non_blocking=True)
            queries = queries.to(device, non_blocking=True)

            query_encodings, key_encodings = model(queries, keys)

            #compute loss and gradient and update encoders
            loss = moco_loss(query_encodings, key_encodings)
            losses[epoch-1].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save model and loss to checkpoint every 5 epochs
        if epoch % 5 == 0:
            if not os.path.exists(modelPath / trainingName):
                os.makedirs(modelPath / trainingName)
            torch.save(model.state_dict(), modelPath / trainingName / f"model_epoch{epoch}.pth")
            torch.save(moco_loss.state_dict(), modelPath / trainingName / f"loss_epoch{epoch}.pth")

        print(f"Epoch {epoch} loss: {sum(losses[epoch - 1]) / len(losses[epoch - 1])} ")
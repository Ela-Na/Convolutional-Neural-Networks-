# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:23:29 2020

@author: Ela
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
import random


def random_seed(seed_value):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False


def dset():
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())
    return train_dataset, test_dataset


class CNN(nn.Module):
    """CREATE MODEL CLASS"""
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1,
                              out_channels=16,
                              kernel_size=3,
                              stride=1,
                              padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=3,
                              stride=1,
                              padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        # FC
        out = self.fc1(out)

        return out


def train(train_loader, test_loader, model, optimizer, criterion):
    iteration = 0
    for epoch in range(num_epochs):
        print("Epoch #", epoch+1, "is now started!")
        epoch_total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            epoch_total_loss += loss.item()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:
                print("Iteration", iteration, "is done")

        # Calculate Accuracy
        correct = 0
        total = 0
        # Iterate through test dataset
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        accuracy = 100 * correct / total

        # Print Loss
        print(f"Epoch: {epoch+1}. "
              f"Loss: {epoch_total_loss}. "
              f"Accuracy: {accuracy}")
    print("total iters:", iteration)

if __name__ == "__main__":
    # seed = 21
    # random_seed(seed)
    train_dataset, test_dataset = dset()  # LOADING DATASET

    batch_size = 100
    num_epochs = 2

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = CNN()
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train(train_loader, test_loader, model, optimizer, criterion)
    # torch.save(model, "./CNN.pt")
    # model2 = torch.load("./CNN.pt")

    # torch.save(model.state_dict(), "./CNN_State.pt")
    # model2 = CNN()
    # model2.load_state_dict(torch.load("./CNN_State.pt"))

    # torch.save({
    #         "Learning Rate": learning_rate,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "batch size": batch_size,
    #         "number of epochs": num_epochs
    #         }, "./checkpoint.pt")
    # checkpoint = torch.load("./checkpoint.pt")
    # model3 = CNN()
    # model3.load_state_dict(checkpoint["model_state_dict"])

    # model.state_dict().keys()
    # model.state_dict()["cnn1.weight"]
    # model.state_dict()["cnn1.weight"].shape
    # train_dataset.data.shape
    # num_epochs*train_dataset.data.shape[0]/batch_size
    # import matplotlib.pyplot as plt
    # plt.imshow(train_dataset.data[0])
    # print(train_dataset.data[0])
    # torch.min(train_dataset.data)
    # torch.max(train_dataset.data)
© 
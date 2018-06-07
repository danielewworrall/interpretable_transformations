from __future__ import print_function
import sys
import time

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from matplotlib import pyplot as plt
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, height, width):
        super(Net, self).__init__()

        self.height = height
        self.width = width
        self.num_mid = int(height*width / 2)

        self.down1 = nn.Linear(self.height*self.width, self.num_mid)
        self.down2 = nn.Linear(self.num_mid, 10)
        self.up2 = nn.Linear(10, self.num_mid)
        self.up1 = nn.Linear(self.num_mid, self.height*self.width)


    def forward(self, x):
        x = F.relu(self.down1(x))
        x = self.down2(x)
        x = F.relu(self.up2(x))
        return self.up1(x)

    def feature_transformer(self, input, params):
        """For now we assume the params are just a single rotation angle

        Args:
            input: [N,c] tensor, where c = 2*int
            params: [N,1] tensor, with values in [0,2*pi)
        Returns:
            [N,c] tensor
        """
        # First reshape activations into [N,c/2,2,1] matrices
        x = input.view(input.size(0),input.size(1)/2,2,1)
        # Construct the transformation matrix
        sin = torch.sin(params)
        cos = torch.cos(params)
        matrix = torch.concat([sin, -cos, cos, sin], )
        # Multiply
        output = transformation @ x
        # Reshape and return
        return output.view(input.size())


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Reshape data
        data = data.view(data.size(0), -1)

        # Forward pass
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        # L1 loss
        loss = torch.mean(torch.sum(torch.abs(output - data), 1))

        # Backprop
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()


def test(args, model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # Reshape data
            data = data.view(data.size(0), -1)

            # Forward pass
            data = data.to(device)
            output = model(data)
            break
        print(output.size())





def view_images(images, nrow=None):
    """View the images in a grid format

    Args:
        images: array of shape [N,rgb,h,w], rgb=1 or 3
    """
    if nrow == None:
        nrow = int(np.floor(np.sqrt(images.size(0))))

    img = torchvision.utils.make_grid(images, nrow=nrow, normalize=True).numpy()
    img = np.transpose(img, (1,2,0))

    plt.imshow(img)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(28,28).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()

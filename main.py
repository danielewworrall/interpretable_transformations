from __future__ import print_function
import os
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
from scipy.ndimage.interpolation import rotate
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, height, width, device):
        super(Net, self).__init__()

        self.height = height
        self.width = width
        self.device = device

        # Init model layers
        self.down1 = nn.Linear(self.height*self.width, 600)
        self.down2 = nn.Linear(600, 400)
        self.down3 = nn.Linear(400, 400)
        self.up3 = nn.Linear(400, 400)
        self.up2 = nn.Linear(400, 600)
        self.up1 = nn.Linear(600, self.height*self.width)


    def forward(self, x, params):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.down3(x)   # Must be linear layer!

        # Feature transform layer
        x = self.feature_transformer(x, params)

        x = F.relu(self.up3(x))
        x = F.relu(self.up2(x))
        return F.sigmoid(self.up1(x))   # Sigmoid output for MNIST


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
        transform = torch.cat([sin, -cos, cos, sin], 1)
        transform = transform.view(transform.size(0),1,2,2).to(self.device)
        # Multiply: broadcasting taken care of automatically
        # [N,1,2,2] @ [N,channels/2,2,1]
        output = torch.matmul(transform, x)
        # Reshape and return
        return output.view(input.size())


def rotate_tensor(input):
    """Nasty hack to rotate images in a minibatch, this should be parallelized
    and set in PyTorch

    Args:
        input: [N,c,h,w] **numpy** tensor
    Returns:
        rotated output and angles in radians
    """
    angles = 2*np.pi*np.random.rand(input.shape[0])
    angles = angles.astype(np.float32)
    outputs = []
    for i in range(input.shape[0]):
        output = rotate(input[i,...], 180*angles[i]/np.pi, axes=(1,2), reshape=False)
        outputs.append(output)
    return np.stack(outputs, 0), angles


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Reshape data
        targets, angles = rotate_tensor(data.numpy())
        targets = torch.from_numpy(targets).to(device)
        targets = targets.view(targets.size(0), -1)
        angles = torch.from_numpy(angles).to(device)
        angles = angles.view(angles.size(0), 1)
        data = data.view(data.size(0), -1)

        # Forward pass
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, angles)

        # Binary cross entropy loss
        loss_fnc = nn.BCELoss(size_average=False)
        loss = loss_fnc(output, targets)

        # Backprop
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()


def test(args, model, device, test_loader, epoch):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # Reshape data: apply multiple angles to the same minibatch, hence
            # repeat
            data = data.view(data.size(0), -1)
            data = data.repeat(args.test_batch_size,1)

            angles = torch.linspace(0, 2*np.pi, steps=args.test_batch_size)
            angles = angles.view(args.test_batch_size, 1)
            angles = angles.repeat(1, args.test_batch_size)
            angles = angles.view(args.test_batch_size**2, 1)

            # Forward pass
            data = data.to(device)
            output = model(data, angles)
            break
        output = output.cpu()
        output = output.view(-1,1,28,28)
        save_images(output, epoch)


def save_images(images, epoch, nrow=None):
    """Save the images in a grid format

    Args:
        images: array of shape [N,1,h,w], rgb=1 or 3
    """
    if nrow == None:
        nrow = int(np.floor(np.sqrt(images.size(0))))

    img = torchvision.utils.make_grid(images, nrow=nrow, normalize=True).numpy()
    img = np.transpose(img, (1,2,0))

    plt.figure()
    plt.imshow(img)
    plt.savefig("./output/epoch{:04d}".format(epoch))
    plt.close()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
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

    # Set up dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Init model and optimizer
    model = Net(28,28,device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Where the magic happens
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)


if __name__ == '__main__':
    # Create save path
    path = "./output"
    if not os.path.exists(path):
        os.makedirs(path)
    main()

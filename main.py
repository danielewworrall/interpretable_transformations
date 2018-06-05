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
        self.mid_shape = np.asarray([16,16])
        self.num_mid = int(np.prod(self.mid_shape))
        self.grid = self.create_grid((28,28))

        self.rc1 = self.create_receptive_fields(self.mid_shape)
        self.fc1 = nn.Linear(self.height*self.width, self.num_mid)
        self.fc2 = nn.Linear(self.num_mid, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def create_receptive_fields(self, shape, sigma=5.):
        """Return regularizers to encourage the formation of receptive fields

        Args:
            num_rc: number of receptive fields to use (int)
            sigma: width of each receptive field (float)
        Returns:
            [num_rc, height*width] tensor of receptive field weights
        """
        cy = np.linspace(0.,self.height, num=shape[1]/2)
        cy = np.stack([cy,cy],0)
        cx = np.linspace(0.,self.width, num=shape[0]/2)
        cx = np.stack([cx,cx],0)
        cX, cY = np.meshgrid(cx, cy)
        centers = np.stack([cX, cY], 0)
        centers = centers[:,:,:,np.newaxis,np.newaxis]

        # Use a soft Gaussian receptive field
        X, Y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        coords = np.stack([X, Y], 0)[:,np.newaxis,np.newaxis,:,:]
        dist2 = np.sum(((coords - centers)/sigma)**2, axis=0)
        dist2 = np.reshape(dist2, [np.prod(shape),-1]).astype(np.float32)
        return torch.from_numpy(dist2)

    def create_grid(self, shape):
        """Create grid of positions

        Args:
            shape: 2d x and y positions (tuple/list)
        Returns:
            [2,x,y] coordinates
        """
        cy = np.arange(shape[1])
        cx = np.arange(shape[0])
        cX, cY = np.meshgrid(cx, cy)
        return torch.from_numpy(np.stack([cX, cY], 0).astype(np.float32))

    def apply_rc_regularizer(self, input, dist2):
        """Apply the receptive field regularization

        Args:
            input: [o,i] tensor
            dist2: [o,i] tensor of squared distances
        Returns:
            scalar loss
        """
        return torch.mean((input*dist2)**2)

    def apply_locality_regularizer(self, input, grid, sigma=3.):
        """Apply a receptive field regularizer with non-fixed center

        Args:
            input: [o,i] tensor
            sigma: scalar width (float)
        Returns:
            scalar loss
        """
        # Compute mean position of activations
        input = input.view(input.size(0),grid.size(1),grid.size(2))
        input = input.unsqueeze(1)
        grid = grid.unsqueeze(0)

        # weights are the input
        weights = torch.abs(input)

        # integrate mean over position, weighted by input
        mean = (weights * grid) / self.iter_sum(weights, 1)
        mean = torch.sum(mean, 2, keepdim=True)
        mean = torch.sum(mean, 3, keepdim=True)

        # integrate squared distance over position, weighted by input
        dist2 = (grid - mean)*(grid - mean)
        dist2 = torch.sum(dist2, 1, keepdim=True)

        spread = (weights * dist2) / self.iter_sum(weights, 1)
        spread = torch.sum(spread, 1)
        spread = torch.sum(spread, 1)
        spread = torch.sum(spread, 1)

        return torch.sum(spread)

    def iter_sum(self, x, channel):
        end = len(x.size())
        for i in range(channel, end):
            x = torch.sum(x, i, keepdim=True)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss += 100.*model.apply_rc_regularizer(model.fc1.weight, model.rc1.to(device))
        #loss += 0.01*model.apply_locality_regularizer(model.fc1.weight, model.grid.to(device))

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
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

    plt.figure(1, figsize=(12,12))
    plt.show(block=False)
    for epoch in range(1, args.epochs + 1):
        view_images(model.fc1.weight.data.cpu().view(-1,1,28,28))
        plt.draw()
        plt.pause(0.001)

        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()

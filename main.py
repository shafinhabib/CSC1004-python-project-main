from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from utils.config_utils import read_args, load_config, Dict2Object

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :param log_file: path to the log file
    :return:
    """
    model.train()
    correct = 0
    loss_value = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Calculate training accuracy and loss
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss_value += loss.item()

    # Calculate training accuracy and loss
    training_acc = 100. * correct / len(train_loader.dataset)
    training_loss = loss_value / len(train_loader)

    # Write to log file
    with open("test.txt", "a") as f:
        f.write(f"Training: epoch {epoch} | loss: {training_loss:.4f} | accuracy: {training_acc:.2f}\n")

    # Print training accuracy and loss
    print(f"Training: epoch {epoch} | loss: {training_loss:.4f} | accuracy: {training_acc:.2f}")

    return training_acc, training_loss


def test(model, device, test_loader, filename, epoch):
    """
    test the model and return the testing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :param filename: the name of the file to which to write the testing results
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        # write the current testing accuracy and loss to the file
        with open(filename, 'a') as f:
            f.write(f"Epoch {epoch}, Testing accuracy: {(100. * correct / len(test_loader.dataset)):.2f}%, "
                    f"Testing loss: {test_loss:.4f}\n")

    testing_acc = 100. * correct / len(test_loader.dataset)
    testing_loss = test_loss
    return testing_acc, testing_loss


def plot(title, xlabel, ylabel, xdata, ydata):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xdata, ydata)
    plt.show()


def run(config):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(config.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        """record training info, Fill your code"""
        test_acc, test_loss = test(model, device, test_loader, epoch)
        """record testing info, Fill your code"""
        scheduler.step()
        """update the records, Fill your code"""

    """plotting training performance with the records"""
    plot("Training Accuracy", "Epoch", "Accuracy (%)", epoches, training_accuracies)
    plot("Training Loss", "Epoch", "Loss", epoches, training_loss)

    """plotting testing performance with the records"""
    plot("Testing Accuracy", "Epoch", "Accuracy (%)", epoches, testing_accuracies)
    plot("Testing Loss", "Epoch", "Loss", epoches, testing_loss)

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def plot(title, xlabel, ylabel, xdata, ydata):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xdata, ydata)
    plt.show()

if __name__ == '__main__':
    arg = read_args()

    """toad training settings"""
    config = load_config(arg)

    """train model and record results"""
    run(config)

    """plot the mean results"""
    # plot_mean()

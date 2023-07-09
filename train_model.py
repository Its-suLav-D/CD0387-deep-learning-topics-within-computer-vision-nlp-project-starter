#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os 
import argparse


#TODO: Import dependencies for Debugging andd Profiling
from smdebug import modes
import smdebug.pytorch as smd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def train(epoch, model, train_loader, criterion, optimizer, hook):  # add 'epoch' as the first argument
    model.train()
    hook.set_mode(smd.modes.TRAIN)

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
def net():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 133))
    return model


def create_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Prepare datasets
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def main(args):
    train_loader, val_loader, test_loader = create_data_loaders(args.data, args.batch_size)
    model = net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(criterion)

    for epoch in range(args.epochs):
        train(epoch, model, train_loader, criterion, optimizer, hook)  # pass 'epoch' as the first argument

        test(model, test_loader, criterion, hook)
    
    torch.save(model, os.path.join(args.model_dir, 'model.pth'))



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
        help="training data path in S3"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="location to save the model to"
    )
    
    
    args=parser.parse_args()
    
    main(args)

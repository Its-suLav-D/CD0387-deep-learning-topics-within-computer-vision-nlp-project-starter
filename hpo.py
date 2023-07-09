# Import your dependencies.
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
import sys


logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Initialize a model
def net():
    logger.info("Initializing the network.")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 133))
    return model


# Training function
def train(model, train_loader, criterion, optimizer, device):
    logger.info("Training the network.")
    model.train() 
    try:
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()
    except Exception as e:
        logger.error("Exception occurred during training: ", exc_info=True)
        raise e



# Testing function
def test(model, test_loader, device):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def validate(model, val_loader, criterion, device):
    logger.info("Validating the network.")
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    average_loss = running_loss / len(val_loader)
    logger.info("Validation Loss: {:.4f}".format(average_loss))
    return average_loss




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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Initialize a model
    model = net()

    logger.info("Defining the criterion and optimizer.")

    
    # Define the criterion and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    
    train_loader, val_loader, test_loader = create_data_loaders(args.data, args.batch_size)

    
    for epoch in range(args.epochs):
        logger.info("Epoch {}/{}".format(epoch+1, args.epochs))
        train(model, train_loader, criterion, optimizer, device)
        validate(model, val_loader, criterion, device)

    
    # Test the model
    accuracy = test(model, test_loader, device)
    logger.info('Accuracy of the network on the test images: %d %%' % accuracy)

    print('Accuracy of the network on the test images: %d %%' % accuracy)
    
    # Save the trained model
    logger.info("Saving the trained model.")

    torch.save(model, os.path.join(args.model_dir, 'model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters sent by the client are passed as command-line arguments to the script.

    
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
        default=os.environ["SM_CHANNEL_TRAINING"],
        help="training data path in S3"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="location to save the model to"
    )
    

    # Parse the arguments
    args = parser.parse_args()

    main(args)

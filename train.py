# Student Name: James Kekong
# python train.py flowers
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import numpy as np
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
import argparse
import time

def main():
    args = get_arguments()
    input_layers = None
    output_size = None
    epochs = None
    lr = None    
    if args.model == 'densenet121':
        input_layers = 1024
        output_size = 102
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                                        ('input',  nn.Linear(1024, 550)),
                                        ('relu1',  nn.ReLU()),
                                        ('dropout1',  nn.Dropout(0.5)),
                                        ('linear2',  nn.Linear(550, 200)),
                                        ('relu2',  nn.ReLU()),
                                        ('linear3', nn.Linear(200, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                       ]))            
        
    elif args.model == 'vgg19':
        input_layers = 25088
        output_size = 102
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                                        ('input',  nn.Linear(25088, 5000)),
                                        ('relu1',  nn.ReLU()),
                                        ('dropout1',  nn.Dropout(0.5)),
                                        ('linear2',  nn.Linear(5000, 500)),
                                        ('relu2',  nn.ReLU()),
                                        ('linear3', nn.Linear(500, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                       ])) 
    
    model.classifier = classifier
    trainloader, validloader, testloader, valid_data, train_data, test_data, train_transforms, test_transforms, valid_transforms = data_parser(args.data_path)
    
    if args.cuda:
        model = model.cuda()
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
     
    train_model(model, trainloader, validloader, criterion=criterion, optimizer=optimizer, epochs=int(args.epochs), cuda=args.cuda)
    validate_model(model, testloader, cuda=args.cuda)
    checkpoint = {'input_size': input_layers,
              'output_size': output_size,
              'epochs': epochs,
              'batch_size': 64,
              'learning_rate': lr,    
              'model': model,
              'classifier': classifier,
              'class_to_idx':  trainloader.dataset.class_to_idx,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict()
             }
    torch.save(checkpoint, 'checkpoint.pth')

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", action="store", dest="save_dir", default="checkpoint.pth" , help = "Set directory to save checkpoints")
    parser.add_argument("--model", action="store", dest="model", default="densenet121" , help = "Set architechture('densenet121' or     'vgg19')")
    parser.add_argument("--learning_rate", action="store", dest="lr", default=0.001 , help = "Set learning rate")
    parser.add_argument("--hidden_units", action="store", dest="hidden_units", default=512 , help = "Set number of hidden units")
    parser.add_argument("--epochs", action="store", dest="epochs", default=5 , help = "Set number of epochs")
    parser.add_argument("--gpu", action="store_true", dest="cuda", default=False , help = "Use CUDA for training")
    parser.add_argument('data_path', action="store")
    
    return parser.parse_args()


def data_parser(data_path):    
    train_dir = data_path + '/train'
    valid_dir = data_path + '/valid'
    test_dir = data_path + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validloader, testloader, valid_data, train_data, test_data, train_transforms, test_transforms, valid_transforms
    
    
def train_model(model, trainloader, validloader, criterion, optimizer, epochs=3, cuda=False):
    start_time = time.time()
    if cuda and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    steps = 0
    print_every = 20
    for e in range(epochs):
        running_loss = 0
        for images, labels in iter(trainloader):
            inputs, targets = images, labels
            steps += 1

            if torch.cuda.is_available():
                model.cuda()
                inputs, targets = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()        
            output = model.forward(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                valid_loss = 0
                for ii, (images, labels) in enumerate(validloader):
                    inputs, labels = images, labels
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = model.forward(inputs)
                        valid_loss += criterion(output, labels).item()
                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()
    elapsed_time = time.time() - start_time
    print('Elapsed Time: {:.0f}m {:.0f}s'.format(elapsed_time//60, elapsed_time % 60))
    
def validate_model(model, testloader, cuda=False):
    model.eval()
    accuracy = 0
    if cuda and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    for ii, (images, labels) in enumerate(testloader):
        inputs = images
        labels = labels
        with torch.no_grad():
            if torch.cuda.is_available():
                model.cuda()
                inputs, labels = inputs.cuda(), labels.cuda()
            output = model.forward(inputs)
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    
main()
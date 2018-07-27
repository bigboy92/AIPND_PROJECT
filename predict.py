# Student Name: James Kekong
# python predict.py flowers/test/102/image_08030.jpg checkpoint.pth

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
import argparse

def main():
    args = get_arguments()
    cuda = args.cuda
    model, class_to_idx, idx_to_class = load_checkpoint(args.checkpoint, cuda)
    
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
      
    prob, classes = predict(args.input, model, 
                                          class_to_idx, 
                                          idx_to_class, 
                                          cat_to_name, 
                                          topk=args.top_k,
                                          cuda= False)
    print(prob)
    print([cat_to_name[x] for x in classes])
    
    print("The prediction is \"{}\" with a certainty of {:.3f}%".format([cat_to_name[x] for x in classes][0], prob[0]*100))
    
def get_arguments():
    """ 
    Retrieve command line keyword arguments
    """
    parser_msg = 'Predict.py takes two command line arguments, \n\t1.The image to be predicted \n\t2. the checkpoint from the trained nerual network'
    parser = argparse.ArgumentParser(description = parser_msg)

    # Manditory arguments
    parser.add_argument("input", action="store")
    parser.add_argument("checkpoint", action="store")

    # Optional arguments
    parser.add_argument("--top_k", action="store", dest="top_k", default=5, help="Number of top results you want to view.")
    parser.add_argument("--category_names", action="store", dest="categories", default="cat_to_name.json", 
                        help="Number of top results you want to view.")
    parser.add_argument("--cuda", action="store_true", dest="cuda", default=False, help="Set Cuda True for using the GPU")

    return parser.parse_args()

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    
    Parameters:
    image()
    
    Returns:
    
    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

def predict(image_path, model, class_to_idx, idx_to_class ,cat_to_name, topk=5, cuda=False):
    if cuda and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
        
    # TODO: Implement the code to predict the class from an image file
    image = None
    model.eval()
    with Image.open(image_path) as img:
        image = process_image(img)
    with torch.no_grad():
        if torch.cuda.is_available():
            model.cuda()
            image = image.cuda()
            
        image = image.unsqueeze(0)
        output = model.forward(image.float())
        ps = torch.exp(output).data.cpu().numpy()[0]
        topk_index = np.argsort(ps)[-topk:][::-1] 
        idx = [idx_to_class[x] for x in topk_index]
        prob = ps[topk_index]
    return prob, idx 

def load_checkpoint(filepath, cuda):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
    idx_to_class = { v : k for k,v in class_to_idx.items()}   
    return model, checkpoint['class_to_idx'], idx_to_class

main()

    
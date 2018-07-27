# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

##PyTorch Transfer Learning
PyTorch transfer learning example developed as part of Udacity's AI Programming with Python Nanodegree program.

Getting Started

Python 3.6.5:

Numpy
PyTorch
TorchVision

CUDA 9.1

##Sample Data
###Download sample data using curl:

curl -O https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
###And extract using tar:

###mkdir flowers
``` $ tar -xvzf flower_data.tar.gz -C flowers ```

## To run the train.py file run the following
``` python train.py flowers ```

## Run the predict.py file by running the following code
``` python predict.py flowers/test/102/image_08030.jpg checkpoint.pth ```
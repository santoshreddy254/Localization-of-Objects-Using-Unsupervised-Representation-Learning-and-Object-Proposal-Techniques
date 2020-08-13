import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import cv2
import io
import requests
from PIL import Image
import pdb
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
import random
import collections
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_score,recall_score, accuracy_score
from utils.helper import calc_accuracy, get_prediction, make_one_hot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.model_zoo import load_url

import torchvision
import torchvision.datasets as dset
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchsummary import summary
from torchviz import make_dot
from utils.dataloader import VOC_Dataloader
from utils import helper
from models.vgg import vgg16
from models.resnet import resnet18
from models.squeezenet import squeezenet1_1
from utils.util import normalize_tensor
from utils.util import remove_layer
from utils.util import replace_layer
from utils.util import initialize_weights

import random
import argparse


# Create the parser
my_parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
my_parser.add_argument('--checkpoint_path',
                       type=str,
                       default="/home/smuthi2s/perl5/RnD-Weakly-Supervised-Object-Localization/Experiments/New_Experiment/weights/Baseline_local_VOC_classification_new_network_2_vgg16_30.pt",
                       help='Path for trained checkpoint')
my_parser.add_argument('--batch_size',
                       type=int,
                       default=32,
                       help='batch size')
my_parser.add_argument('--dataset',
                       type=str,
                       default = "VOC",
                       help='Have option of three datsets VOC, YCB, at_work')
my_parser.add_argument('--dataset_path',
                       type=str,
                       default = '/scratch/smuthi2s/VOC/',
                       help='Give the path to dataset')
my_parser.add_argument('--backbone',
                       type=str,
                       default = "vgg16",
                       help='Have option of three backbones vgg16, resnet18, squeezenet1_1')

args = my_parser.parse_args()
batch_size = args.batch_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset = args.dataset
if dataset == "VOC":
    root= args.dataset_path
    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
    n_classes = len(object_categories)
    trainset = VOC_Dataloader(root = root, mode='train', dataaug=True,
                                       size=(256, 256), scales=(0.75, 1., 1.25, 1.5, 1.75, 2.))

    valset = VOC_Dataloader(root = root, mode='val', dataaug=True,
                                       size=(256, 256), scales=(0.75, 1., 1.25, 1.5, 1.75, 2.))
elif dataset == "YCB":
    root_dir = args.dataset_path
    transform = transforms.Compose(
                                   [transforms.Resize([256, 256]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
    trainset = ImageFolder(root = root_dir+"train/",transform=transform)
    n_classes = len(trainset.classes)
    object_categories = trainset.classes
    valset = ImageFolder(root = root_dir+"val/",transform=transform)
elif dataset == "at_work":
    root_dir = args.dataset_path
    transform = transforms.Compose(
                                   [transforms.Resize([256, 256]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
    trainset = ImageFolder(root = root_dir+"train/",transform=transform)
    n_classes = len(trainset.classes)
    object_categories = trainset.classes
    valset = ImageFolder(root = root_dir+"val/",transform=transform)

print("Length of traning set: ",len(trainset))
print("Length of validation set: ",len(valset))
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = args.backbone
if model_name == "vgg16":
    model = vgg16(architecture_type="cam",pretrained=True, num_classes = n_classes, large_feature_map=False)
elif model_name == "resnet18":
    model = resnet18('cam',num_classes = n_classes)
elif model_name == "squeezenet1_1":
    model = squeezenet1_1(num_classes = n_classes)

print(model)

checkpoint = torch.load(args.checkpoint_path,map_location="cpu")
model.load_state_dict(checkpoint,strict=True)
exper_name = dataset+"_classification_new_network_"+model_name
print("Results for check point: ",args.checkpoint_path)
print("results identify with name: ",exper_name,)
accuracy = 0
precision = 0
recall = 0
for i,[image,label] in enumerate(val_loader):
    x = Variable(image).to(device)
    y_= Variable(label).to(device)
    if list(label.size()) == [batch_size] or len(list(label.size())) == 1:
        label = make_one_hot(label,n=n_classes)
    pred = model(image, label)
    pred = helper.get_prediction(pred)
    pred = pred.cpu().data.numpy().astype(np.bool)
    label = label.cpu().data.numpy().astype(np.bool)
    accuracy += calc_accuracy(pred,label,batch_size)
    precision += precision_score(label,pred,average="samples")
    recall += recall_score(label,pred,average="samples")

print("Validation accuracy: ",accuracy/(i+1))
print("Precision: ",precision/(i+1))
print("Recall: ",recall/(i+1))

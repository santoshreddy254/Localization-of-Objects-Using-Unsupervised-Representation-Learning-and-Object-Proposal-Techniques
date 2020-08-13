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
my_parser.add_argument('--batch_size',
                       type=int,
                       default=32,
                       help='batch size')
my_parser.add_argument('--save_after',
                       type=int,
                       default = 5,
                       help='Save weights after every n epochs')

my_parser.add_argument('--num_epochs',
                       type=int,
                       default = 10,
                       help='number of epochs')
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
my_parser.add_argument('--experiment_number',
                       type=str,
                       default = "1",
                       help='Give unique identity to experiment')
# Execute the parse_args() method
args = my_parser.parse_args()



batch_size = args.batch_size
learning_rate = 0.0001
num_epoch = args.num_epochs


# ## 3) Dataloader

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


params = model.parameters()
optimizer = torch.optim.SGD(params, lr= 0.1)
criterion = nn.BCELoss(reduction='mean')



exper_name = dataset+"_classification_16_16_"+model_name
print("Identify the weights with name: ",exper_name)
save_after = 10
for ep in range(num_epoch):
    total_loss = 0.0
    acc = 0
    for batch_idx, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)
        if list(target.size()) == [batch_size] or len(list(target.size())) == 1:
            target = helper.make_one_hot(target,n=n_classes)
        pred = model(images, target)
        loss = criterion(pred, target)
        total_loss += loss.item()
        pred = helper.get_prediction(pred)
        pred = pred.cpu().data.numpy().astype(np.bool)
        target = target.cpu().data.numpy().astype(np.bool)
        acc+=(helper.calc_accuracy(pred,target,batch_size) * 100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_average = total_loss / (batch_idx+1)
    classification_acc = acc / (batch_idx+1)
    print('EPOCH : {:03d}/{:3d} | Loss : {:.4f}, training_accuracy: {:.4f}'.format(ep+1, num_epoch, loss_average, classification_acc))
    if (ep+1)%save_after == 0:
        torch.save(model.state_dict(), '/home/smuthi2s/perl5/RnD-Weakly-Supervised-Object-Localization/Experiments/New_Experiment/weights/Baseline_local_{}_{}.pt'.format(exper_name, ep+1))
# writer.close()

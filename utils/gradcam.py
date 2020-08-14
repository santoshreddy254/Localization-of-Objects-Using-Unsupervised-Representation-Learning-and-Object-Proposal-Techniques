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
from utils.gradcam import *
import random
import argparse
import time
from utils.util import t2n
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, pre_features, features, target_layers):
        self.model = model
        self.pre_features = pre_features
        self.features = features
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        for pref in self.pre_features:
            x = getattr(self.model, pref)(x)

        submodel = getattr(self.model, self.features)
        # go through the feature extractor's forward pass
        for name, module in submodel._modules.items():
            # print(name, module)
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x
class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model,
                 pre_feature_block=[],
                 feature_block='features',
                 target_layers='35',
                 classifier_block=['classifier']):
        self.model = model
        self.classifier_block = classifier_block
        # assume the model has a module named `feature`      ⬇⬇⬇⬇⬇⬇⬇⬇
        self.feature_extractor = FeatureExtractor(self.model,
                                                  pre_feature_block,
                                                  feature_block,
                                                  target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        # ⬇ target layer    ⬇ final layer's output
        target_activations, output = self.feature_extractor(x)
        print('target_activations[0].size: {}'.format(target_activations[0].size()))# for vgg'35 ([1, 512, 14, 14])
        print('output.size: {}'.format(output.size()))                              # for vgg'36 ([1, 512, 7, 7])

        for i, classifier in enumerate(self.classifier_block):
            if i == len(self.classifier_block) - 2:
                output = output.view(output.size(0), -1)
                print('output.view.size: {}'.format(output.size()))                 # for vgg'36 ([1, 25088])

            output = getattr(self.model, classifier)(output)
            print('output.size: {}'.format(output.size()))                          # for vgg'36 ([1, 1000])

        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img =         np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, pre_feature_block, feature_block, target_layer_names, classifier_block, use_cuda):
        self.model = model
        self.model.eval()
        self.pre_feature_block = pre_feature_block
        self.feature_block = feature_block
        self.classifier_block = classifier_block
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model,
                                      pre_feature_block,
                                      feature_block,
                                      target_layer_names,
                                      classifier_block)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

#         with open('imagenet_class_index.json') as f:
#             labels = json.load(f)
#         labels
#         print('prediction[{}]: {}'.format(index, labels[str(index)][1]))
        print('output.size: {}'.format(output.size()))                            # for vgg'36 ([1, 1000])
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        #print('output: {}'.format(output))
        #print('one_hot: {}'.format(one_hot))                                      #
        getattr(self.model, self.feature_block).zero_grad()
        for classifier in self.classifier_block:
            getattr(self.model, classifier).zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = self.extractor.get_gradients()
        #print('len(gradients): {}'.format(len(gradients)))
        print('gradients[0].size(): {}'.format(gradients[0].size()))
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        print('target.size(): {}'.format(target.size()))
        target = target.cpu().data.numpy()[0, :]
        print('target.shape: {}'.format(target.shape))

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        print('weights.shape: {}'.format(weights.shape))
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        print('cam.shape: {}'.format(cam.shape))            # (14, 14)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        #print('cam: {}'.format(cam))
        print('cam.shape: {}'.format(cam.shape))
        cam = np.maximum(cam, 0)                            # remove negative numbers
        cam = cv2.resize(cam, (256, 256))
        print('cam.shape: {}'.format(cam.shape))
        #print('cam: {}'.format(cam))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
def preprocess_image_gradCAM(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img =         np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input_img = preprocessed_img.requires_grad_(True)
    return input_img

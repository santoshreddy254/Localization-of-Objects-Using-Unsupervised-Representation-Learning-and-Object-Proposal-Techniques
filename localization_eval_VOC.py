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
from utils.helper import *


# Create the parser
my_parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
my_parser.add_argument('--checkpoint_path',
                       type=str,
                       default="/home/smuthi2s/perl5/RnD-Weakly-Supervised-Object-Localization/Experiments/New_Experiment/weights/Baseline_local_VOC_classification_new_network_2_vgg16_30.pt",
                       help='Path for trained checkpoint')
my_parser.add_argument('--wsol_method',
                       type=str,
                       default='cam',
                       help='select cam or gradCAM')
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
batch_size = 1
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









checkpoint = torch.load(args.checkpoint_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
root= args.dataset_path+'/VOC2012/ImageSets/Segmentation/'
mode='val'
filename = [x[:-1] for x in open(root+mode+'.txt')]
count = 0
loc_accuracy_30=0
loc_accuracy_50=0
loc_accuracy_70=0
t = 0
cor_pred = False
cam_threshold_list = list(np.round(np.arange(0, 1, 0.1),2))

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.ToTensor(),
   normalize
])

num_bins = len(cam_threshold_list) + 2
threshold_list_right_edge = np.append(cam_threshold_list,
                                           [1.0, 2.0, 3.0])
gt_true_score_hist = np.zeros(num_bins, dtype=np.float)
gt_false_score_hist = np.zeros(num_bins, dtype=np.float)

final_thresh_list = []
img_to_bbox_ratio = []


config = {
        'vgg19':    {
            'pre_feature': [],
            'features': 'features',
            'target': ['35'],
            'classifier': ['classifier']
        },
        'resnet18': {
            'pre_feature': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3'],
            'features': 'layer4',
            'target': ['1'],
            'classifier': ['avgpool', 'fc']
        },
        'vgg16':    {
            'pre_feature': [],
            'features': 'features',
            'target': ['29'],
            'classifier': ['conv6','relu','avgpool','fc','sigmoid']
        },
            'squeezenet1_1':    {
            'pre_feature': [],
            'features': 'features',
            'target': ['12'],
            'classifier': ['classifier']
        }
    }


model_name = args.backbone
cam_generation = args.wsol_method
config = config[model_name]
grad_cam = GradCam(model,
                       pre_feature_block=config['pre_feature'],
                       feature_block=config['features'], # features
                       target_layer_names=config['target'],
                       classifier_block=config['classifier'],  # classifier
                       use_cuda=False)

for fn in filename:
    img_path = args.dataset_path+"/VOC2012/JPEGImages/{}.jpg".format(fn)
    img_pil = Image.open(img_path)
    seg_img_path = args.dataset_path+"/VOC2012/SegmentationClass/{}.png".format(fn)
    seg_img = np.array(Image.open(seg_img_path))
    an_path = args.dataset_path+'/VOC2012/Annotations/{}.xml'.format(fn)
    ann = helper.parse_voc_xml(ET.parse(an_path).getroot())
    ann = ann['annotation']['object']
    if not isinstance(ann, list):
        ann = [ann]
    label = torch.zeros((20,), dtype=torch.float)
    for bbox in ann:
        label[object_categories.index(bbox['name'])] = 1
    start = time.time()
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = model(img_variable, label)
#     logit = output_dict['logits']
    h_x = logit.data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    img = cv2.imread(img_path, 1)
    img = np.float32(cv2.resize(img, (256, 256))) / 255
    input_gradCAM = preprocess_image_gradCAM(img)
    for a in ann:
        if a['name'] == object_categories[idx[0]]:
            cor_pred = True
            temp = [a['bndbox']['xmin'],a['bndbox']['ymin'],a['bndbox']['xmax'],a['bndbox']['ymax']]
            bbox_real = [int(i) for i in temp]
            break
    if cor_pred:
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        img_to_bbox_ratio.append((bbox_real[2]-bbox_real[0])*(bbox_real[3]-bbox_real[1])*100/(height*width))
        temp_mask = np.zeros((height,width))
        pix_loc = np.where(seg_img==idx[0]+1)
        temp_mask[pix_loc] = 255
        gt_mask = temp_mask/255
        output = model(img_variable, idx[0], return_cam=True)
        feature_maps = t2n(output[0])
        fc_weights = t2n(output[1])
        CAMs = t2n(output[-1])
        if cam_generation == "gradCAM":
            gradCAM_mask = grad_cam(img_variable, idx[0])
            heatmap = cv2.resize(gradCAM_mask,(width, height))
            heatmap = np.array(heatmap,dtype=np.float64)
        else:
            heatmap = cv2.resize(CAMs[0],(width, height))
            heatmap = normalize_scoremap(np.array(heatmap,dtype=np.float64))
        if not np.isnan(heatmap).any():
            estimated_boxes_at_each_thr, number_of_box_list = compute_bboxes_from_scoremaps(heatmap,cam_threshold_list,True)
            max_iou_bboxes = [[[0,0,0,0],0,0] for i in range(len(cam_threshold_list))]
            for p,i in enumerate(estimated_boxes_at_each_thr):
                gt_iou = 0
                for j in i:
                    iou = helper.bb_intersection_over_union(j,bbox_real)
                    if iou >= gt_iou:
                        max_iou_bboxes[p][0] = j
                        max_iou_bboxes[p][1] = iou
                        max_iou_bboxes[p][2] = cam_threshold_list[p]
                        gt_iou = iou
            max_iou_bboxes.sort(key = lambda x: x[1],reverse=True)
            image_bbox = draw_bboxes(img,max_iou_bboxes,bbox_real)

            final_thresh_list.append(max_iou_bboxes[0][2])
            gt_true_scores = heatmap[gt_mask == 1]
            gt_false_scores = heatmap[gt_mask == 0]

            # histograms in ascending order
            gt_true_hist, _ = np.histogram(gt_true_scores,
                                           bins=threshold_list_right_edge)
            gt_true_score_hist += gt_true_hist.astype(np.float)

            gt_false_hist, _ = np.histogram(gt_false_scores,
                                            bins=threshold_list_right_edge)
            gt_false_score_hist += gt_false_hist.astype(np.float)
            end = time.time()
            t+=(end-start)
            iou = helper.bb_intersection_over_union(max_iou_bboxes[0][0],bbox_real)
            if iou>=0.5:
                loc_accuracy_50+=1
            if iou>=0.3:
                loc_accuracy_30+=1
            if iou>=0.7:
                loc_accuracy_70+=1
            if count<10:
                for j in range(0, 3):
                    print('{:.3f} -> {}'.format(probs[j], object_categories[idx[j]]))
                # print(max_iou_bboxes[0])
                # plt.title(object_categories[idx[0]]+" - iou: "+str(round(iou,2)))
                # plt.imshow(cv2.cvtColor(image_bbox, cv2.COLOR_BGR2RGB))
                # plt.savefig("new_vgg16_output_"+str(count)+".png")
                # plt.show()
                # plt.title("CAM")
                # plt.imshow(heatmap*255)
                # plt.savefig("new_vgg16_cam_"+str(count)+".png")
                # plt.show()
            count+=1
            cor_pred=False


def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])


auc = get_PxAP(gt_true_score_hist,gt_false_score_hist)


print("Top-1 localization accuracy iou=0.3: ",loc_accuracy_30/count)
print("Top-1 localization accuracy iou=0.5: ",loc_accuracy_50/count)
print("Top-1 localization accuracy iou=0.7: ",loc_accuracy_70/count)
print("Top-1 localization accuracy ",((loc_accuracy_30/count)+(loc_accuracy_50/count)+(loc_accuracy_70/count))/3)

print("Time per image: ",t/count)
print("Mask AUC {}".format(auc))

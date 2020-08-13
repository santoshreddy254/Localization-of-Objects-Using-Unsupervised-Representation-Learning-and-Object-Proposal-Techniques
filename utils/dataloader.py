import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
import random
import collections
import xml.etree.ElementTree as ET
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VOC_Dataloader(Dataset):
    def __init__(
        self,
        root,
        mode='train_aug',
        dataaug=True,
        size=(256,256),
        scales=(0.75, 1, 1.25, 1.5, 1.75, 2.)
    ):

        self.classes=(
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )
        self.root = root
        self.mode = mode
        self.dataaug = dataaug
        if dataaug:
            self.size = size
            self.scales = scales
            self.ColorJitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)

        self.filename = [x[:-1] for x in open(root+'/VOC2012/ImageSets/Main/'+mode+'.txt')]
        self.len = len(self.filename)

    def Normalize(self, img, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        img = np.array(img, np.float32)
        img = np.array(img) /255.
        img[:,:,0] = (img[:,:,0] - mean[0])/std[0]
        img[:,:,1] = (img[:,:,1] - mean[1])/std[1]
        img[:,:,2] = (img[:,:,2] - mean[2])/std[2]

        return torch.from_numpy(img).permute(2,0,1)

    def RandomScale(self, img, scales=(0.75, 1, 1.25, 1.5, 1.75, 2.)):
        scale = random.choice(scales)
        w, h = img.size
        w, h = int(scale*w), int(scale*h)

        return img.resize((w,h), Image.BILINEAR)

    def RandomCrop_and_Resize(self, img, size=(256,256)):
        imgW, imgH = img.size

        cropratio = random.uniform(0.9 , 1)
        cropH, cropW = cropratio*imgH, cropratio*imgW

        sw, sh = abs(imgW - cropW), abs(imgH - cropH)
        crop = (int(sw), int(sh), int(sw) + cropW, int(sh) + cropH)

        return img.crop(crop).resize(size, Image.BILINEAR)

    def RandomHFlip(self, img):
        if random.random() < 0.5:
            img = img.transpose(0)
        return img

    def __getitem__(self, index):
        fn = self.filename[index]
        img_path = self.root+'/VOC2012/JPEGImages/{}.jpg'.format(fn)
        img = Image.open(img_path)

        if self.dataaug:
            # Random Scale
            img = self.RandomScale(img, scales=self.scales)

            # Random Crop
            img = self.RandomCrop_and_Resize(img, size=self.size)

            # H_Flip
            img = self.RandomHFlip(img)

            # Color Jitter
            img = self.ColorJitter(img)

        img = self.Normalize(img)

        # Load Annotation
        an_path = self.root+'/VOC2012/Annotations/{}.xml'.format(fn)
        ann = self.parse_voc_xml(ET.parse(an_path).getroot())
        ann = ann['annotation']['object']
        if not isinstance(ann, list):
            ann = [ann]

        label = torch.zeros((20,), dtype=torch.float)
        gt_bbox = torch.Tensor([[0, 0, 0, 0] for i in range(20)])
        for bbox in ann:
            label[self.classes.index(bbox['name'])] = 1
            bounding_box = [int(bbox['bndbox']['xmin']),int(bbox['bndbox']['ymin']),int(bbox['bndbox']['xmax']),int(bbox['bndbox']['ymax'])]
            gt_bbox[self.classes.index(bbox['name'])] = torch.FloatTensor(bounding_box)
        return img, label

    def __len__(self):
        return self.len

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

import os
import random
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import cv2
import sys
import json

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def parse_berry(a,scale_factor):
    berry_targets = []
    for berry_target in a:
        berry_targets.append((np.array(berry_target['segmentation']).astype(float)*scale_factor).astype(int).tolist())
    return berry_targets

def letterbox(im, probmap, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        probmap = cv2.resize(probmap, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    probmap = cv2.copyMakeBorder(probmap, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return im, np.expand_dims(probmap,axis=2), ratio, (dw, dh)

def augment_hsv(im, hgain=0.015, sgain=0.7, vgain=0.4):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

class ProbmapDataset(Dataset):
    def __init__(self, train = False, base_dir = None, data_split = None, probmap_type = None):
        with open(os.path.join(base_dir,data_split)) as f:
            image_names = []
            lines = f.readlines()
            for line in lines:
                name = line.replace('\n','')
                image_names.append(name)
        if train:
            image_names = image_names*4
            random.shuffle(image_names)
        self.nSamples = len(image_names)
        self.image_names = image_names
        self.train = train
        self.train = train
        self.base_dir = base_dir
        self.probmap_type = probmap_type

    def _crop(self, img, prob_target):        
        h, w = img.shape[:-1]
        crop_size = (384, 384)
        
        dx = int(random.random() * (w - crop_size[0]))
        dy = int(random.random() * (h - crop_size[1]))
        
        img = img[dy:crop_size[0] + dy, dx:crop_size[1] + dx,:]
        prob_target = prob_target[dy:crop_size[0] + dy, dx:crop_size[1] + dx]
        return img, prob_target
    
    def _scale(self, img, prob_target):
        if random.random() > 0.5:
            scale_factor = 0.8 + 0.4 * random.random()
            w, h = img.shape[:-1]
            w_new = int(w * scale_factor)
            h_new = int(h * scale_factor)
            if scale_factor>=1:
                ip = cv2.INTER_CUBIC
            else:
                ip = cv2.INTER_AREA
            img = cv2.resize(img, (h_new,w_new), interpolation=ip)
            prob_target = cv2.resize(prob_target, (h_new, w_new), interpolation=ip)
        
        return img, prob_target
    
    def _flip(self, img, prob_target):
        if random.random() > 0.5:
            prob_target = np.fliplr(prob_target)
            img = np.fliplr(img)  
        if random.random() > 0.5:
            prob_target = np.flipud(prob_target)
            img = np.flipud(img) 
        return img, prob_target
    
    def _align(self, img, prob_target):
        H = int((img.shape[0] + 32 - 1) / 32) * 32
        W = int((img.shape[1] + 32 - 1) / 32) * 32
        img, prob_target, ratio, pad = letterbox(img, prob_target, (H,W), auto=False, scaleup=True)
        
        return img, prob_target

    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        img_path = os.path.join(self.base_dir,'images',self.image_names[index])
        img = cv2.imread(img_path)
        prob_path = img_path.replace('.jpg','.h5').replace('images','probmaps_'+self.probmap_type)
        prob_file = h5py.File(prob_path, 'r')
        prob_target = np.asarray(prob_file['probmap'])
        img = cv2.resize(img,(prob_target.shape[1],prob_target.shape[0]),interpolation=cv2.INTER_AREA)

        if self.train == True:
            augment_hsv(img)
            img, prob_target = self._crop(img, prob_target)
            img, prob_target = self._scale(img, prob_target)
            img, prob_target = self._flip(img, prob_target)
        img, prob_target = self._align(img, prob_target)

        img = img/255
        img = np.transpose(img,(2,0,1))
        prob_target = np.transpose(prob_target,(2,0,1))
        if self.train:
            return img, prob_target
        else:
            return img, prob_target, np.asarray(prob_file['count']), self.image_names[index]

    
if __name__ == '__main__':
    base_dir = sys.path[0]
    train = True
    dataset = ProbmapDataset(train=train, base_dir=os.path.join(base_dir, 'datasets', 'GBISC'), data_split='train.txt', probmap_type='mask')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    if train:
        for i, (train_data, probmap) in enumerate(data_loader):
            print(train_data.shape,torch.max(train_data),probmap.shape,torch.max(probmap))
            plt.subplot(121)
            plt.imshow(np.transpose(train_data[0].cpu().numpy(),(1,2,0))[..., ::-1])
            plt.subplot(122)
            plt.imshow(np.transpose(probmap[0].cpu().numpy(),(1,2,0)))
            plt.show(block=True)
            if i == 50:
                break
    else:
        for i, (train_data, probmap, num, _) in enumerate(data_loader):
            print(train_data.shape,torch.max(train_data),probmap.shape,torch.max(probmap),num)
            plt.subplot(121)
            plt.imshow(np.transpose(train_data[0].cpu().numpy(),(1,2,0))[..., ::-1])
            plt.subplot(122)
            plt.imshow(np.transpose(probmap[0].cpu().numpy(),(1,2,0)))
            plt.show(block=True)
            if i == 50:
                break
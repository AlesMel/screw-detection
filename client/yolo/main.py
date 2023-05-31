import os
import requests
import random
import json
import numpy as np
from skimage import io
from scipy import ndimage
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms

datapath = "D:\\FEI-STU\\TP\\network\\client\\dataset"
with open(os.path.join(datapath, 'screws.json')) as f:
  data = json.load(f)

from matplotlib import pyplot as plt
from matplotlib import rcParams, gridspec
from matplotlib import patches, transforms as plt_transforms

rcParams['figure.figsize'] = [16, 6]
rcParams['font.size'] =14
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True

def unpack_bbox(bbox):
  #bbox as in the json/COCO data format (centerx, centery, width, height, theta is in radians)

  rot_center = np.array((bbox[1], bbox[0])).T
  width = bbox[3]
  height = bbox[2]
  theta = -bbox[4]+np.pi/2 #radians
  return rot_center, width, height, theta


def rotcorners_from_coords(rot_center, width, height, theta):
  rotation = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta))))

  wvec = np.dot(rotation, (width/2, 0))
  hvec = np.dot(rotation, (0, height/2))
  corner_points = rot_center + [wvec+hvec, wvec-hvec, -wvec+hvec, -wvec-hvec]
  return corner_points


def rotbbox_from_coords(rot_center, width, height, theta):
  corner_points = rotcorners_from_coords(rot_center, width, height, theta)
  rot_bbox = np.array((corner_points.min(0), corner_points.max(0))).astype(np.int)
  #constrain inside image
  rot_bbox[rot_bbox < 0] = 0

  return rot_bbox


def extract_subimg_bbox(im, bbox):
  return extract_subimg(im, *unpack_bbox(bbox))


def extract_subimg(im, rot_center, width, height, theta):
  rot_bbox = rotbbox_from_coords(rot_center, width, height, theta)

  subimg = im[rot_bbox[0,1]:rot_bbox[1,1],rot_bbox[0,0]:rot_bbox[1,0]]
  rotated_im = ndimage.rotate(subimg, np.degrees(theta)+180)
  newcenter = (np.array(rotated_im.shape)/2).astype(np.int)
  rotated_im = rotated_im[int(newcenter[0]-height/2):int(newcenter[0]+height/2), int(newcenter[1]-width/2):int(newcenter[1]+width/2), :3]  #drop alpha channel, if it's there

  return rotated_im

print(data.keys())
print(data['images'][0])
print(data['annotations'][0])

imgdir = os.path.join(datapath, 'images')

#remap images to dict by id
imgdict = {l['id']:l for l in data['images']}
#read in all images, can take some time
for i in imgdict.values():
  i['image'] = io.imread(os.path.join(imgdir, i['file_name']))[:, :,: 3]  # drop alpha channel, if it's there

# remap annotations to dict by image_id
from collections import defaultdict
annodict = defaultdict(list)
for annotation in data['annotations']:
  annodict[annotation['image_id']].append(annotation)

# setup list of categories
categories = data['categories']
ncategories = len(categories)
cat_ids = [i['id'] for i in categories]
category_names = {7:'nut', 3:'wood screw', 2:'lag wood screw', 8:'bolt',
                  6:'black oxide screw', 5:'shiny screw', 4:'short wood screw',
                  1:'long lag screw', 9:'large nut', 11:'nut', 10:'nut',
                  12:'machine screw', 13:'short machine screw' }

# Let's look at one image and it's associated annotations
imageid = 100
im = imgdict[imageid]['image']
gs = gridspec.GridSpec(1, 1 + len(annodict[imageid]),
                       width_ratios=[1,]+[.1]*len(annodict[imageid]),
                       wspace=.05)
plt.figure()
ax = plt.subplot(gs[0])
plt.imshow(im)
cmap_normal = plt.Normalize(0, ncategories)

for i, annotation in enumerate(annodict[imageid]):
  bbox = annotation['bbox']

  # plt.scatter(*rot_center)
  # plt.scatter(*corner_points.T, c='r')

  ax = plt.subplot(gs[0])
  color = plt.cm.jet(cmap_normal(annotation['category_id']))
  rect = patches.Rectangle((bbox[1] - bbox[3]/2 ,
                            bbox[0] - bbox[2]/2), bbox[3], bbox[2],
                           linewidth=1, edgecolor=color, facecolor='none')
  t = plt_transforms.Affine2D().rotate_around(bbox[1], bbox[0], -bbox[4]+np.pi/2)
  rect.set_transform(t + plt.gca().transData)
  ax.add_patch(rect)

  plt.subplot(gs[i + 1])
  rotated_im = extract_subimg_bbox(im, bbox)
  plt.imshow(rotated_im)
  plt.axis('off')
  plt.title(annotation['category_id'])

plt.colorbar(ticks=range(ncategories), label='category')
plt.clim(-0.5, ncategories - .5)
plt.show()
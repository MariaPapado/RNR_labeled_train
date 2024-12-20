import torch
import numpy as np
import os
import dataloader
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import shutil
from torchvision.models import resnet50, ResNet50_Weights
from myresnet import ModifiedResNet
import imutils
from test_funcs import *
import matplotlib.pyplot as plt


#####
dataset_path = '/home/mariapap/DATASETS/gasunie_cycles/CYCLE4/test/'
#######

keep_ids = os.listdir(dataset_path + 'T1/')

print(len(keep_ids))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NumClasses = 4
Net = resnet50(weights=ResNet50_Weights.DEFAULT)
modified_resnet = ModifiedResNet(Net)

modified_resnet.load_state_dict(torch.load('./saved_models/net_26.pt', weights_only=True))  ###@@@@@@ this gave gasune 14 14
modified_resnet.eval()
modified_resnet.to(device)

c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)

psize=128
step=128
area_patch=psize*psize

for id in keep_ids:
    im1 = Image.open(dataset_path + 'T1/{}'.format(id))
    im2 = Image.open(dataset_path + 'T2/{}'.format(id))
    label = Image.open(dataset_path + 'MASK/{}'.format(id))

    if '_r' in id:
        lu = 2
    elif '_nr' in id:
        lu = 0
    cat = lu

    im1, im2, label = np.array(im1), np.array(im2), np.array(label)

    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Example: get the first contour and calculate its area
    contour = contours[0]  # Access the first contour (or loop through contours if multiple)
    area = cv2.contourArea(contour)
    #print('area', area)
    if area>0:
        if area<=area_patch:
            x, y, w, h = centered_patch(contour, psize)
            #print('x', 'y', x, y)
            patches1, patches2 = cut_patches(im1, im2, label, [x], [y], psize, 0)
            #plotit(label, x, y, w, h)

        else:
            x, y, w, h = cv2.boundingRect(contour)
            #print('x', 'y', x, y, w, h)
            xs, ys = tile_label(y, x, w, h, psize, step)
            #print('xy ys', xs, ys)
            patches1, patches2 = cut_patches(im1, im2, label, xs, ys, psize, 0)
            #plotit(label, x, y, w, h)
    tpi_probs = [0]*NumClasses
    for p in range(0, len(patches1)):
        p1, p2 = patches1[p], patches2[p]
        p1, p2 = imutils.normalize_img(p1), imutils.normalize_img(p2)
        p1, p2 = np.transpose(p1, (2, 0, 1)), np.transpose(p2, (2, 0, 1))
        p1, p2 = torch.Tensor(p1).float().unsqueeze(0).to(device), torch.Tensor(p2).float().unsqueeze(0).to(device)        
        patch = torch.cat((p1,p2), 1)
#        print(patch.shape)
        pred = modified_resnet(patch)
        soft_pred = F.softmax(pred, dim=1).data.cpu().numpy()[0]
        Prob_,Pred_=pred.max(dim=1)
        for t in range(len(tpi_probs)):
            tpi_probs[t] +=soft_pred[t]

    if len(patches1)!=0:
        tpi_probs = [t/len(patches1) for t in tpi_probs]
    #print(tpi_probs)
        max_element = max(tpi_probs)

# Get the index of the maximum element
        max_index = tpi_probs.index(max_element)

        c_matrix = c_matrix + confusion_matrix([cat], [max_index], labels=np.arange(NumClasses))
    #else:
    #    print(id, area, len(contour), xs, ys)
print('TEST_c_matrix')
print(c_matrix)
print(sum(c_matrix))
#    Prob_,Pred_=pred.max(dim=1)

  #  c_matrix = c_matrix + confusion_matrix(label.data.cpu().numpy(), Pred_.data.cpu().numpy(), labels=np.arange(NumClasses))










    #    print(x, y, h ,w) #h| w-
'''
        if patches1 and patches2:
            #print('PPPPPPPPPPPPPPPPPPPPPPPPPPP')
            save_patches(patches1, patches2, lu, './PATCHES/{}'.format(split), id)

    


    im1, im2 = imutils.normalize_img(im1), imutils.normalize_img(im2)

    im1, im2 = np.transpose(im1, (2, 0, 1)), np.transpose(im2, (2, 0, 1))
    im1, im2 = torch.Tensor(im1).float().unsqueeze(0), torch.Tensor(im2).float().unsqueeze(0)





    
'''



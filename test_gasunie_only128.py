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
from myresnet_mask import ModifiedResNet
import imutils
from test_funcs import *


#####
dataset_path = '/home/mariapap/DATASETS/gasunie_cycles/CYCLE4/test/'
#######

keep_ids = os.listdir(dataset_path + 'T1/')

print(len(keep_ids))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NumClasses = 4
Net = resnet50(weights=ResNet50_Weights.DEFAULT)
modified_resnet = ModifiedResNet(Net)

modified_resnet.load_state_dict(torch.load('./saved_models/net_23.pt', weights_only=True))  ###@@@@@@ this gave gasune 14 14
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
        lu = 3
    elif '_nr' in id:
        lu = 1
    cat = lu

    im1, im2, label = np.array(im1), np.array(im2), np.array(label)
    idx255 = np.where(label==255)
    label[idx255]=1
    #print('immmmmmmmm', im1.shape)

    im1, im2 = imutils.normalize_img(im1), imutils.normalize_img(im2)
    im1, im2 = np.transpose(im1, (2, 0, 1)), np.transpose(im2, (2, 0, 1))
    im1, im2, mask = torch.Tensor(im1).float().unsqueeze(0).to(device), torch.Tensor(im2).float().unsqueeze(0).to(device), torch.Tensor(label).float().unsqueeze(0).unsqueeze(0).to(device)         
    im = torch.cat((im1,im2), 1)
#        print(patch.shape)
    pred = modified_resnet(im, mask)
    soft_pred = F.softmax(pred, dim=1).data.cpu().numpy()[0].tolist()
    max_element = max(soft_pred)
    max_index = soft_pred.index(max_element)
#    if max_index==0:
#        if np.abs(soft_pred[0]-soft_pred[2])<0.05:
#            max_index = 2

    if max_index==1:
        if np.abs(soft_pred[2]- soft_pred[1])<0.2 or np.abs(soft_pred[3]- soft_pred[1])<0.2:
            max_index = 3

    c_matrix = c_matrix + confusion_matrix([cat], [max_index], labels=np.arange(NumClasses))


    im1 = Image.open(dataset_path + 'T1/{}'.format(id))
    im2 = Image.open(dataset_path + 'T2/{}'.format(id))
    label = Image.open(dataset_path + 'MASK/{}'.format(id))

    im1.save('./CHECKS/im1/{}'.format(id))
    im2.save('./CHECKS/im2/{}'.format(id))
    label.save('./CHECKS/label/{}_{}.png'.format(id[:-4], max_index))

    #else:
    #    print(id, area, len(contour), xs, ys)




#            img1 = Image.open('/home/mariapap/DATASETS/RNR/relevant_not_relevant/R_NR/before/{}'.format(data_idx[0]))
#            img2= Image.open('/home/mariapap/DATASETS/RNR/relevant_not_relevant/R_NR/after/{}'.format(data_idx[0]))
#            mask = Image.open('/home/mariapap/DATASETS/RNR/relevant_not_relevant/R_NR/mask/{}'.format(data_idx[0]))

print('TEST_c_matrix')
print(c_matrix)
#print(sum(c_matrix))
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



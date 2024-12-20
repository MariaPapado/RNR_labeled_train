import rasterio
import pickle
from tqdm import tqdm
import os
from PIL import Image as imagepil
from shapely import geometry
import rasterio.features
import numpy as np
import imutils
import torch
from torchvision.models import resnet50, ResNet50_Weights
from myresnet_mask import ModifiedResNet
from test_funcs import *
import cv2
from test_funcs import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import shutil

preds_folder = 'PREDS'
if os.path.exists(preds_folder):
    shutil.rmtree(preds_folder)

# Create the directory
os.mkdir(preds_folder)
os.mkdir('./PREDS/R_wrong')

os.mkdir('./PREDS/R_wrong/before')
os.mkdir('./PREDS/R_wrong/after')
os.mkdir('./PREDS/R_wrong/mask')


test_directory = '/home/mariapap/DATASETS/dataset_FP_v0/fa_benchmark_test/'
#test_directory = '/home/mariapap/DATASETS/fa_benchmark_test_ngc/'
filenames = os.listdir(test_directory)

NumClasses = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Net = resnet50(weights=ResNet50_Weights.DEFAULT)
modified_resnet = ModifiedResNet(Net)

#modified_resnet.load_state_dict(torch.load('./saved_models/net_14.pt', weights_only=True))  ###@@@@@@ this gave gasune 14 14
modified_resnet.load_state_dict(torch.load('./saved_models_v1_w05/net_15.pt', weights_only=True))  ###@@@@@@ this gave gasune 14 14

modified_resnet.eval()
modified_resnet.to(device)

c_matrix = np.zeros((2,2), dtype=int)

psize=128
step=128
area_patch=psize*psize
omitted = []
width, height = 512, 512
missed_events = []
for filename in tqdm(filenames):
    data_tpi = pickle.load(open(test_directory+filename, 'rb'))
    #print(data_tpi)

    if data_tpi['tpi']['notes'] == 'good':
        data_tpi['tpi']['notes'] = 'relevant'
        cat  = 1
    if data_tpi['tpi']['notes'] != 'relevant':
        cat  = 0
        data_tpi['tpi']['notes'] = 'not relevant'

#    if data_tpi['tpi']['confirmation_factor']!=2:
#        cat = 0
#    else:
#        cat = 1


    tpi_poly = geometry.Polygon(data_tpi['tpi']['coordinates'])
    geotiff_transform = rasterio.transform.from_bounds(*data_tpi['poly'].bounds, width, height)
    mask_valid = rasterio.features.rasterize([tpi_poly], out_shape=(width, height), transform=geotiff_transform).astype(np.uint8)
    binary_mask = 255*(mask_valid>0).astype(np.uint8)
    print(np.unique(binary_mask))
    idx255 = np.where(binary_mask==255)
    binary_mask[idx255]=1

    image_before = data_tpi['image_before']
    image_after  = data_tpi['image_after']

#    print(image_before.shape)

    im1, im2, label = np.array(image_before), np.array(image_after), np.array(binary_mask)
    idx255 = np.where(label==255)
    label[idx255]=1
###########################################################################################################################
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Example: get the first contour and calculate its area
    patches1, patches2, masks = [], [], []
    for contour in contours:
        #contour = contours[0]  # Access the first contour (or loop through contours if multiple)
        area = cv2.contourArea(contour)
        #print('area', area)
        if area>0:
            if area<=area_patch:
                x, y, w, h = centered_patch(contour, psize)
                #print('x', 'y', x, y)
                patches1, patches2, masks = cut_patches(im1, im2, label, [x], [y], psize, 0, patches1, patches2, masks)
                #plotit(label, x, y, w, h)
            else:
                x, y, w, h = cv2.boundingRect(contour)
                #print('x', 'y', x, y, w, h)
                xs, ys = tile_label(y, x, w, h, psize, step)
                #print('xy ys', xs, ys)
                patches1, patches2, masks = cut_patches(im1, im2, label, xs, ys, psize, 0, patches1, patches2, masks)
                #plotit(label, x, y, w, h)
    tpi_probs = [0]*NumClasses
    for p in range(0, len(patches1)):

        mask = masks[p] #/255.
#        print(np.unique(mask))
        idxn0 = np.where(mask!=0)
        mask[idxn0] = 1

        p1, p2 = patches1[p], patches2[p]
        p1, p2 = imutils.normalize_img(p1), imutils.normalize_img(p2)
        p1, p2 = np.transpose(p1, (2, 0, 1)), np.transpose(p2, (2, 0, 1))
        p1, p2, mask = torch.Tensor(p1).float().unsqueeze(0).to(device), torch.Tensor(p2).float().unsqueeze(0).to(device), torch.Tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)         
        patch = torch.cat((p1,p2), 1)
#        print(patch.shape)
        pred = modified_resnet(patch, mask)
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
#########################################
        if max_index==0:
            if np.abs(tpi_probs[0]-tpi_probs[2])<0.1:
                max_index = 2

#        if max_index==0:
#            if tpi_probs[0]<0.5:
#                max_index=2
#            else:
#                if np.abs(tpi_probs[0]-tpi_probs[2])<0.1:
#                    max_index = 2


#############################################3
        if max_index==1:
            if np.abs(tpi_probs[1]-tpi_probs[2])<0.12 or np.abs(tpi_probs[1]-tpi_probs[3])<0.12:
                max_index = 3
#############################
        if max_index==0 or max_index==1:
            max_index=0
        else:
            max_index=1


        c_matrix = c_matrix + confusion_matrix([cat], [max_index], labels=np.arange(2))


        if cat!=max_index:
            if cat==1:
                image_before = Image.fromarray(image_before)
                image_after = Image.fromarray(image_after)
                binary_mask = Image.fromarray(binary_mask*255)

                image_before.save('./PREDS/R_wrong/before/{}'.format(filename[:-3] + '.png'))
                image_after.save('./PREDS/R_wrong/after/{}'.format(filename[:-3] + '.png'))
                binary_mask.save('./PREDS/R_wrong/mask/{}'.format(filename[:-3] + '.png'))
                missed_events.append(tpi_probs)


    else:
        omitted.append(filename)

        
print('TEST_c_matrix')
print(c_matrix)
print(sum(c_matrix))

print('oooooooooooooooooooo')
print(missed_events)
cm = c_matrix

classes = ['NON-R', 'R']  # Class names

# Plot confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()

# Labeling ticks and axes
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Display counts in cells
thresh = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.title('CONF_MATRX2')
plt.show()

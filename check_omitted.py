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



test_directory = '/home/mariapap/DATASETS/dataset_FP_v0/fa_benchmark_test/'
filenames = os.listdir(test_directory)

omit_list = ['data_tpi_Gasunie-DE-2024_3796.p', 'data_tpi_Gasunie-DE-2024_3650.p', 'data_tpi_Gasunie-DE-2024_2725.p', 'data_tpi_Gasunie-DE-2024_1415.p', 'data_tpi_Gasunie-DE-2024_1370.p']

width, height = 512, 512

for filename in tqdm(filenames):

    if filename in omit_list:
        data_tpi = pickle.load(open(test_directory+filename, 'rb'))
        print('ok')
    #    if data_tpi in omit_list:


        tpi_poly = geometry.Polygon(data_tpi['tpi']['coordinates'])
        geotiff_transform = rasterio.transform.from_bounds(*data_tpi['poly'].bounds, width, height)
        mask_valid = rasterio.features.rasterize([tpi_poly], out_shape=(width, height), transform=geotiff_transform).astype(np.uint8)
        binary_mask = 255*(mask_valid>0).astype(np.uint8)
        idx255 = np.where(binary_mask==255)

        image_before = data_tpi['image_before']
        image_after  = data_tpi['image_after']

        im1, im2, label = np.array(image_before), np.array(image_after), np.array(binary_mask)
        idx255 = np.where(label==255)
        label[idx255]=1
    ###########################################################################################################################
        contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))


'''

        image_before = Image.fromarray(image_before)
        image_after = Image.fromarray(image_after)
        binary_mask = Image.fromarray(binary_mask)

        image_before.save('./check/im1/{}'.format(filename[:-2] + '.png'))
        image_after.save('./check/im2/{}'.format(filename[:-2] + '.png'))
        binary_mask.save('./check/mask/{}'.format(filename[:-2] + '.png'))

'''


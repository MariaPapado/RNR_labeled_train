import torch
import numpy as np
import os
import dataloader
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import shutil
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from myresnet_mask import ModifiedResNet
from torch.autograd import Variable
import torchvision
#from focalloss import FocalLoss


dataset_path = '/home/mariapap/DATASETS/dataset_FP_v1/PATCHES/'


with open("/home/mariapap/DATASETS/dataset_FP_v1/lists/train_list.txt", "r") as file:
    train_ids = [line.strip() for line in file.readlines()]
#(train_ids)

with open("/home/mariapap/DATASETS/dataset_FP_v1/lists/val_list.txt", "r") as file:
    val_ids = [line.strip() for line in file.readlines()] 

trainset = dataloader.RNR_data(dataset_path, train_ids, 'train')
valset = dataloader.RNR_data(dataset_path, val_ids, 'val')

trainloader = DataLoader(trainset, batch_size=8, shuffle=True, drop_last=True)
valloader = DataLoader(valset, batch_size=4, shuffle=True, drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'




#Net=Net.Net(NumClasses=NumClasses,UseGPU=UseCuda) # Create main resnet image classification brach
NumClasses = 4
Net = resnet50(weights=ResNet50_Weights.DEFAULT)
#print(Net)

# Initialize the modified model
modified_resnet = ModifiedResNet(Net)

modified_resnet.to(device)
#print(modified_resnet)

weight_tensor=torch.FloatTensor(4)

#v1_w05
weight_tensor[0]= 0.10
weight_tensor[1]= 0.20
weight_tensor[2]= 0.35
weight_tensor[3]= 0.35

weight_tensor[0]= 0.14
weight_tensor[1]= 0.35
weight_tensor[2]= 0.31
weight_tensor[3]= 0.21


weight_tensor.to(device)

#criterion = FocalLoss()
criterion = nn.CrossEntropyLoss(weight_tensor)
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()

criterion.to(device)
#optimizer = torch.optim.Adam(Net.parameters(), lr=0.0001) 14 14
optimizer = torch.optim.Adam(modified_resnet.parameters(), lr=0.000003)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
epochs = 30

c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)
iter_ = 0

save_folder = 'saved_models'
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)

# Create the directory
os.mkdir(save_folder)




for epoch in range(0, epochs):
    modified_resnet.train()
    train_loss = []
    correct = 0
    total = 0
    for i, batch in enumerate(tqdm(trainloader)):
        img, mask, label, data_idx = batch
        img, mask, label = img.float().to(device), mask.float().to(device), label.to(device)
        #print(img.shape)

        label_1hot = torch.nn.functional.one_hot(label, num_classes=NumClasses)
        #print(img.shape, mask.shape, label)


        optimizer.zero_grad()
        pred = modified_resnet(img, mask)
        Prob_,Pred_=pred.max(dim=1)

        #loss = criterion(pred, label_1hot.float())
        loss = criterion(pred, label_1hot.float())
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    
        c_matrix = c_matrix + confusion_matrix(label.data.cpu().numpy(), Pred_.data.cpu().numpy(), labels=np.arange(NumClasses))

        iter_ += 1

        if iter_ % 100 == 0:
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, epochs, i, len(trainloader),100.*i/len(trainloader), loss.item()))

    print('TRAIN_c_matric')
    print(c_matrix)
    print('TRAIN_mean_loss: ', np.mean(train_loss)) 



    modified_resnet.eval()    
    c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)
    with torch.no_grad():
        val_loss = []
        for i, batch in enumerate(tqdm(valloader)):
            img, mask, label, data_idx = batch
            img, mask, label = img.float().to(device), mask.float().to(device), label.to(device)
            label_1hot = torch.nn.functional.one_hot(label, num_classes=NumClasses)

            pred = modified_resnet(img, mask)
            Prob_,Pred_=pred.max(dim=1)
            loss = criterion(pred, label_1hot.float())

            val_loss.append(loss.item())

            c_matrix = c_matrix + confusion_matrix(label.data.cpu().numpy(), Pred_.data.cpu().numpy(), labels=np.arange(NumClasses))


    print('VAL_c_matric')
    print(c_matrix)
    print('VAL_mean_loss: ', np.mean(val_loss)) 

    torch.save(modified_resnet.state_dict(), './' + save_folder + '/net_{}.pt'.format(epoch))



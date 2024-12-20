import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F




# Example: Adding a custom intermediate layer after layer2 of ResNet
class ModifiedResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()
        # Keep the original layers up to the layer you want to modify
        #self.features = nn.Sequential(
        #    *list(original_model.children())[:1]  # Up to layer2
        #)

        # Add your custom intermediate layer
        #self.attention_layer = nn.Conv2d(65, 1, kernel_size=3, padding=1).cuda()  # Add a new conv layer
        # Continue with the remaining layers from the original model
        #self.remaining_layers = nn.Sequential(
        #    *list(original_model.children())[1:-1],  # After layer2 up to the pooling layer
        #)

#        self.conv1 = original_model.conv1
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        self.avgpool = original_model.avgpool
        self.fc1 = original_model.fc


        # Replace the final fully connected (fc) layer if needed
        self.fc2 = nn.Linear(1000, 500).cuda()  # For example, for 10 output classes
        self.fc3 = nn.Linear(500,256).cuda()
        self.fc4 = nn.Linear(256,4).cuda()

    def process_mask(self, mask, mysize):

        mask = F.interpolate(mask, size=mysize, mode='nearest')  # Resize the mask

        idx0 = torch.where(mask==0)
        idx1 = torch.where(mask==1)

        mask[idx0] = 0.5
        mask[idx1]=1    

        return mask

    def forward(self, x, mask):
        mask32 = self.process_mask(mask, (32,32))
        mask64 = self.process_mask(mask, (64,64))
        mask64 = self.process_mask(mask, (64,64))

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x) #downsampling!!!!
#        att_map = x*mask64

#        x = self.maxpool(x) #downsampling

        x = self.layer1(x) #[8, 256, 16, 16]
        x = self.layer2(x) #[8, 512, 8, 8]
        att_map = x*mask32
        x = self.layer3(att_map) #[8, 1024, 4, 4]    ####sthn arxh only this!!! gia saved_models_v1_w05
        x = self.layer4(x) # [8, 2048, 2, 2]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

# Initialize the modified model
#modified_resnet = ModifiedResNet(Net)
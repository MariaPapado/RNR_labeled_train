import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

# Example: Adding a custom intermediate layer after layer2 of ResNet
class ModifiedResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()

        # Add a new layer at the beginning
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Use the original ResNet layers except the final FC
        self.features = nn.Sequential(*list(original_model.children())[1:-1])  # Start from layer after initial conv
        
        # Additional FC layers at the end
        self.fc1 = nn.Linear(2048, 1024)  # First additional FC layer
        self.fc2 = nn.Linear(1024, 512)   # Second additional FC layer
        self.fc3 = nn.Linear(512, 256)    # Third additional FC layer
        
        # Final classification layer (adjust output classes as needed)
        self.final_fc = nn.Linear(256, 4)  # Example for 10 output classes


    def forward(self, x):

        x = self.conv1(x)
        x = self.features(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.final_fc(x)
        #x = F.softmax(x, dim=1)
        return x

# Initialize the modified model
#modified_resnet = ModifiedResNet(Net)

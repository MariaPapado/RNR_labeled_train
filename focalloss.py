import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, alpha=[0.15, 0.18, 0.39, 0.29], gamma=2,reduction='none'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)

        # Apply alpha if provided (for class weights)
        if self.alpha is not None:
            at = torch.tensor(self.alpha, device=input.device)[target.long()]
            ce_loss = at * ce_loss

        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss.mean()
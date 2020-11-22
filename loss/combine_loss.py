import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.dice_loss import DiceLoss
from cross_entropy import  CrossentropyLoss, TopKLoss

class BCEPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus bce
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(BCEPlusDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):

        assert isinstance(predict,list)
        assert isinstance(target,list)
        assert len(predict) == len(target) and len(predict) == 2

        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict[1],target[1])

        bce = nn.BCEWithLogitsLoss(self.weight)
        bce_loss = bce(predict[0],target[0])
        
        total_loss = bce_loss + dice_loss

        return total_loss


class CEPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus cross entropy
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(CEPlusDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        ce = CrossentropyLoss(weight=self.weight,ignore_index=self.ignore_index)
        ce_loss = ce(predict,target)
        
        total_loss = ce_loss + dice_loss

        return total_loss


class TopKCEPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus topk cross entropy 
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(TopKCEPlusDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):

        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        topk = TopKLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        topk_loss = topk(predict,target)
        
        total_loss = topk_loss + dice_loss

        return total_loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class FloatCrossEntropy(nn.Module):
    """Cross Entropy loss for float label

    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(FloatCrossEntropy, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        cross_entropy = nn.CrossEntropyLoss(self.weight,ignore_index=self.ignore_index)  
        return cross_entropy(predict,torch.argmax(target, 1))
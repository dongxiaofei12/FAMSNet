import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class MobileNet(nn.Module):

    def __init__(self, arg1):
        super().__init__()
        self.arg1 = arg1
        print(self.arg1)
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self, pretrained=None):
        pass
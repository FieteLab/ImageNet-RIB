import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
from torchvision.datasets import ImageFolder

class ImageNetV2(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        root = os.path.join(root, 'imagenetv2-matched-frequency-format-val')
        super(ImageNetV2, self).__init__(root, transform, target_transform)
        self.change_label()
    
    def change_label(self):
        for i in range(len(self.samples)):
            path = self.samples[i][0]
            label = int(path.split('/')[-2])
            self.samples[i] = (self.samples[i][0], label)
            self.targets[i] = label
        self.class_to_label = {str(label):label for label in range(1000)}
        self.classes = list(self.class_to_label.keys())

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class textCNN(nn.Module):
    def __init__(self, num_classes):
        super(textCNN, self).__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(2816, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        logit = self.sigmoid(x)
        return logit

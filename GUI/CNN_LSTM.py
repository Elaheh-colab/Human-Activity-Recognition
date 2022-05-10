import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Softmax, Module, BatchNorm2d, Dropout, LeakyReLU, Sequential
from torch.nn.init import kaiming_uniform_, constant_, xavier_uniform_
from torchvision import transforms, datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.optim import SGD
import sys
from torch import save, load, cuda
from torch import device
import os
import torch.nn.functional as F
import math
import time
import torch.optim 
import torchvision.models as models

# Defining as global the device to use (by default CPU).
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN_LSTM(nn.Module):
    #We should have batch_size*seq_length batches everytime. So, the input is (batch_size*seq_length, C=3, H, W). 
    #After extracting features from model_CNN the output shape is (batch_size*seq_length, C_new, H_new, W_new).
    #We reshape the output to (batch_size, seq_length, -1) for passing to model_RNN
    def __init__(self, input_size=2048*1*1, hidden_size=4, num_layers=2, seq_length=20, video_batch_size=16, num_classes=50):
        self.seq_length = seq_length
        self.video_batch_size = video_batch_size
        super(CNN_LSTM, self).__init__()

        self.model_CNN = models.resnext101_32x8d(pretrained=True)
        for param in self.model_CNN.parameters():
            param.requires_grad = False
        
        # x -> (batch_size, seq_length, input_size) --> seq_length=hidden_size
        # nn.LSTM(input_size, hidden_size, num_layers) --> hidden_size=seq_length, num_layers=how many LSTM layer we want to stack together
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        #since we used bidirectional=True in LSTM so the num_layers should be twice or num_layers*2
        self.h0 = torch.zeros(num_layers*2, video_batch_size, hidden_size).to(device=device)
        self.c0 = torch.zeros(num_layers*2, video_batch_size, hidden_size).to(device=device)

        #if bidirectional=True the output of LSTM will be hidden_size*2
        # self.fc1 = nn.Linear(hidden_size*2, 512)
        self.fc1 = nn.Linear(hidden_size*2, 256)
        self.drp1 = nn.Dropout(0.5)
        kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        constant_(self.fc1.bias, 0)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.drp2 = nn.Dropout(0.4)
        kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        constant_(self.fc2.bias, 0)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(256, num_classes)
        xavier_uniform_(self.fc3.weight)
        constant_(self.fc2.bias, 0)
        
    def forward(self, x):
        # with torch.no_grad():
        self.CNN_Feature_Out = {}
        self.model_CNN.avgpool.register_forward_hook(self.get_activation('Feature_Out'))
        self.model_CNN(x)
        out_CNN = self.CNN_Feature_Out['Feature_Out']
        # print(out_CNN.shape)
        out_CNN = out_CNN.reshape(self.video_batch_size, self.seq_length, -1)
        # print(out_CNN.shape)
        out_lstm, _ = self.lstm(out_CNN, (self.h0,self.c0))
        #out_lstm -> (batch_size, seq_length, hidden_size) -> we only need the last output in our sequence not middle outputs -> out_lstm[:, -1, :]

        # x = self.drp1(self.act1(self.fc1(out_lstm[:, -1, :])))
        # x = self.drp2(self.act2(self.fc2(x)))
        x = self.drp1(self.act1(self.fc1(out_lstm[:, -1, :])))
        x = self.fc3(x)

        return(x)
        # return(out_lstm)
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.CNN_Feature_Out[name] = output.detach()
        return hook
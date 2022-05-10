# -*- coding: utf-8 -*-
"""Elaheh_DL_Proj_Preprocessing_9_Categories.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18qUaECw0sZ0SZ0n1p1IMAwDf-kev07wf
"""

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
from matplotlib import pyplot as plt
import cv2
import glob
from PIL import Image
from torchvision.utils import save_image

path_train = "/content/drive/MyDrive/Elaheh/Deep_Learning_Project/Dataset/train"
path_valid = "/content/drive/MyDrive/Elaheh/Deep_Learning_Project/Dataset/test"
path_train_vid = "/content/drive/MyDrive/Elaheh/Deep_Learning_Project/Dataset_Vid_50/train"
path_valid_vid = "/content/drive/MyDrive/Elaheh/Deep_Learning_Project/Dataset_Vid_50/test"

clip_n_frames = 50
clip_time = 5
count_videos = 0

valid_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

i=0
count_video = 0
for path_category in glob.glob(path_train + '/*'):
    print(path_category)
    category = path_category.split("/")[-1]
    for path in glob.glob(path_category + '/*'):
        # print(path)
        vidcap = cv2.VideoCapture(path)
          
        # count the number of frames
        frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        # print(f'fps:{fps}, frames:{frames}')

        count_videos += 1
        
        vidcap = cv2.VideoCapture(path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        frames = []
        
        time = []
        count = 0  # control to have the same number of frames
        count_fps = 0
        count_sections = 0
        while success:
            
            success, image = vidcap.read()
            count += 1
            if(type(image).__module__ == np.__name__):
              new_image = valid_transform(Image.fromarray(image))
              frames.append(new_image)
              count_fps += 1

        num_frames = len(frames)
        video_time = num_frames/fps
        
        if(num_frames < 15):
          print(num_frames)
          continue

        if num_frames < clip_n_frames:
          while (count > 0 and count <= clip_n_frames):
            frames.append(new_image)    # if the number of frames is lower than the num_frames, repeat the last image to reach num_frames
            count +=1
            count_fps += 1
            
          video_file = torch.zeros((3, 224, 224*clip_n_frames))
          for i in range(clip_n_frames):
            video_file[:,:,i*224:(i+1)*224] = frames[i]

          name_video = category + str(count_video) + ".png"
          path_directory = os.path.join(path_train_vid, category)
          if not os.path.exists(path_directory):
              os.makedirs(path_directory)
          path_video = os.path.join(path_directory, name_video)
          save_image(video_file, path_video)

          # print(f'count_video:{count_video}, frames:{len(frames)}, images:{frames[0].shape}')
          count_video += 1

        else:
          num_sections = math.ceil(video_time/clip_time)
          if num_sections == 0:
            num_sections = 1
          frame_rate = int(num_frames/(clip_n_frames*num_sections))
          while (frame_rate == 0):
            num_sections -= 1
            frame_rate = int(num_frames/(clip_n_frames*num_sections))

          while (count_sections < num_sections):
            videos_3d = frames[count_sections*clip_n_frames*frame_rate:(count_sections+1)*clip_n_frames*frame_rate:frame_rate]
            count_sections += 1
              
            video_file = torch.zeros((3, 224, 224*clip_n_frames))
            for i in range(clip_n_frames):
              video_file[:,:,i*224:(i+1)*224] = videos_3d[i]

            name_video = category + str(count_video) + ".png"
            path_directory = os.path.join(path_train_vid, category)
            if not os.path.exists(path_directory):
                os.makedirs(path_directory)
            path_video = os.path.join(path_directory, name_video)
            save_image(video_file, path_video)

            count_video += 1

i=0
count_video = 0
for path_category in glob.glob(path_valid + '/*'):
    print(path_category)
    category = path_category.split("/")[-1]
    for path in glob.glob(path_category + '/*'):
        # print(path)
        vidcap = cv2.VideoCapture(path)
          
        # count the number of frames
        frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        # print(f'fps:{fps}, frames:{frames}')

        count_videos += 1
        
        vidcap = cv2.VideoCapture(path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        frames = []
        
        time = []
        count = 0  # control to have the same number of frames
        count_fps = 0
        count_sections = 0
        while success:
            
            success, image = vidcap.read()
            count += 1
            if(type(image).__module__ == np.__name__):
              new_image = valid_transform(Image.fromarray(image))
              frames.append(new_image)
              count_fps += 1

        num_frames = len(frames)
        video_time = num_frames/fps
        
        if(num_frames < 20):
          print(num_frames)
          continue

        if num_frames < clip_n_frames:
          while (count > 0 and count <= clip_n_frames):
            frames.append(new_image)    # if the number of frames is lower than the num_frames, repeat the last image to reach num_frames
            count +=1
            count_fps += 1
            
          video_file = torch.zeros((3, 224, 224*clip_n_frames))
          for i in range(clip_n_frames):
            video_file[:,:,i*224:(i+1)*224] = frames[i]

          name_video = category + str(count_video) + ".png"
          path_directory = os.path.join(path_valid_vid, category)
          if not os.path.exists(path_directory):
              os.makedirs(path_directory)
          path_video = os.path.join(path_directory, name_video)
          save_image(video_file, path_video)

          # print(f'count_video:{count_video}, frames:{len(frames)}, images:{frames[0].shape}')
          count_video += 1

        else:
          num_sections = math.ceil(video_time/clip_time)
          if num_sections == 0:
            num_sections = 1
          frame_rate = int(num_frames/(clip_n_frames*num_sections))
          while (frame_rate == 0):
            num_sections -= 1
            frame_rate = int(num_frames/(clip_n_frames*num_sections))

          while (count_sections < num_sections):
            videos_3d = frames[count_sections*clip_n_frames*frame_rate:(count_sections+1)*clip_n_frames*frame_rate:frame_rate]
            count_sections += 1
              
            video_file = torch.zeros((3, 224, 224*clip_n_frames))
            for i in range(clip_n_frames):
              video_file[:,:,i*224:(i+1)*224] = videos_3d[i]

            name_video = category + str(count_video) + ".png"
            path_directory = os.path.join(path_valid_vid, category)
            if not os.path.exists(path_directory):
                os.makedirs(path_directory)
            path_video = os.path.join(path_directory, name_video)
            save_image(video_file, path_video)

            count_video += 1

#Separating one folder of image or video to train and test
# from os import walk

# # import OS module
# import os
# import shutil

# # Get the list of all files and directories
# mypath = "/content/drive/MyDrive/Elaheh/Deep_Learning_Project/train/walk/"
# train_path = "/content/drive/MyDrive/Elaheh/Deep_Learning_Project/Dataset/train/Walking/"
# test_path = "/content/drive/MyDrive/Elaheh/Deep_Learning_Project/Dataset/test/Walking/"
# dir_list = os.listdir(mypath)

# print("Files and directories in '", mypath, "' :")

# # prints all files
# FileNames = []
# for file in dir_list:
#     FileNames.append(file)
#     # print(file)

# len_dataset = len(FileNames)
# len_train = int(len_dataset * 0.9)
# len_test = len_dataset-len_train

# if not os.path.exists(train_path):
#     os.mkdir(train_path)
# if not os.path.exists(test_path):
#     os.mkdir(test_path)

# for i in range(len_dataset):
#     if i < len_train:
#         path = os.path.join(mypath + FileNames[i])
#         shutil.copy(path, train_path)
#     else:
#         path = os.path.join(mypath + FileNames[i])
#         shutil.copy(path, test_path)


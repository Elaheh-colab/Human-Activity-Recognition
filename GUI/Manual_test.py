import numpy as np
import keras
import sklearn as sk
import cv2
from keras.applications.resnet import ResNet101, preprocess_input
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LSTM, ConvLSTM2D, Reshape, Conv2D, Dropout, BatchNormalization
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.wrappers import TimeDistributed
import glob
import os
import tensorflow as tf
import json
from keras.models import model_from_json
# import pytictoc
import time
#%matplotlib qt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from threading import Thread
from torchvision import transforms, datasets
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
import torch.nn as nn
from PIL import Image
from CNN_LSTM import CNN_LSTM

img = None
Text_on_Image = ""
Predict_Score = ""
preprocess_enabled = True
network_name = ""
model_is_loaded = False
flag_load_next = True
flag_sync = True

image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

image_transform2 = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

class preprocess_manual:
    def __init__(self, path_to_ckpt, clip_time, video_path):
        global model_is_loaded
        model_is_loaded = False
                
        self.video_path = video_path
        self.clip_time = clip_time
        self.clip_n_frames = 20

        
        self.videos_2d = []
        self.time_2d = []
        self.labels_2d = []



        ##### Real Time
        self.threshold = 0.4

        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self):
        global img
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(img, axis=0)
        # Actual detection.

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})


        im_height, im_width,_ = img.shape
        
        scores = np.array(scores)[0]
        ind_human = np.where(np.array(classes[0])==1)[0]
        scores_human = scores[ind_human]
        boxes_human = boxes[0][ind_human]
        exist_human = scores_human >= self.threshold
        
        
        
        # num_boxes = 3
        if len(exist_human)>=1:
            
            if (exist_human[0]):
                cv2.rectangle(img,(int(boxes_human[0,1] * im_width),
                        int(boxes_human[0,0]*im_height),
                        int(boxes_human[0,3] * im_width),
                        int(boxes_human[0,2]*im_height)),(255,0,0),2)
                
                print("Human Score: ", scores_human[0])
            else:
                print("No Human")
        else:
            exist_human = False
            return exist_human
          
        if len(exist_human)>=2:
            
            if (exist_human[1]):
                cv2.rectangle(img,(int(boxes_human[1,1] * im_width),
                        int(boxes_human[1,0]*im_height),
                        int(boxes_human[1,3] * im_width),
                        int(boxes_human[1,2]*im_height)),(255,0,0),2)
        

        if len(exist_human)>=3:
                
            if (exist_human[2]):
                cv2.rectangle(img,(int(boxes_human[2,1] * im_width),
                        int(boxes_human[2,0]*im_height),
                        int(boxes_human[2,3] * im_width),
                        int(boxes_human[2,2]*im_height)),(255,0,0),2)
        
                
        return exist_human[0]
        

    def manual_prediction(self, network_model, preprocessing):
        global img,new_image,Text_on_Image,Predict_Score, preprocess_enabled, network_name, model_is_loaded, flag_load_next, flag_sync
        network_name = network_model
        preprocess_enabled = preprocessing
        self.running = True
        
        vidcap = cv2.VideoCapture(self.video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("fps:", fps)

        # count the number of frames
        self.num_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("total_frames: ", self.num_frames)

        # self.num_frames = int(fps*self.clip_time)
 
        # self.frame_rate = int(self.num_frames/self.clip_n_frames)
        
        
        self.video_rate = self.num_frames/self.clip_n_frames
        
        # self.frame_rate = 5
        
        success, image_raw = vidcap.read()
        
        new_image = cv2.resize(image_raw, (224,224), interpolation = cv2.INTER_AREA)
        img = cv2.resize(image_raw, (800, 600))

        start_time = time.time()
        exist_human = self.processFrame()
        end_time = time.time()
   
        human_det_delay = end_time - start_time
            
        ### Define Modified Frame Rate Based on human_det_delay and 1/self.frame_rate
        # FPS_HUM_RATIO = (1/self.video_rate)/human_det_delay
        FPS_HUM_RATIO = self.video_rate
        self.frame_rate = max(int(FPS_HUM_RATIO), 1)
        print("Delay: ", human_det_delay, " -> ", "Frame Rate: ", self.frame_rate)
                    
        
        Thread(target = self.prediction).start()
        counter = self.frame_rate
        counter_total = 0
        while(True):
            if model_is_loaded:
                # if flag_load_next:
                if flag_sync:
                    flag_sync2 = True
                if flag_sync2 and flag_load_next:
                    # new_image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
                    # new_image = cv2.resize(image, (224,224))
                    # img = cv2.resize(image, (1280, 720))
                    img = cv2.resize(image_raw, (800, 600))

                    # Wait for 25ms
                    if cv2.waitKey(int(human_det_delay*10)) & 0xFF == ord('q'):
                        self.running = False
                        break
                
                    cv2.putText(img, Text_on_Image + " " + Predict_Score, (100,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,255), thickness=3)
                    #cv2.putText(img, Predict_Score, (100,200), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,255), thickness=3)
                    cv2.imshow('Camera', img)
                    

                    if counter == self.frame_rate:
                        new_image = image_raw
                        if flag_sync == False:
                            flag_sync2 = False
                        flag_sync = False
                        counter = 0
                    else:
                        success, image_raw = vidcap.read()
                        if not success:
                            # print("Not success")
                            self.running = False
                            break
                    
                        # print(counter)
                        counter += 1

                    # counter_total += 1
                    # print("counter_total:", counter_total)
                # else:
                #     if not flag_load_next:
                #         print(f'flag_sync:{flag_sync}, flag_load_next:{flag_load_next}')
            else:
                    # Wait for 25ms
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
        print("FINISHI!!!")
                
    def prediction(self):
        global img,new_image,Text_on_Image,Predict_Score, preprocess_enabled, network_name, model_is_loaded, flag_load_next, flag_sync
        #output of avgpool = 2048*1*1 and output of layer4 is 2048*7*7
        input_size=2048*1*1
        hidden_size=512
        num_layers=2 #how many lstm stack together

        ImageBatchSize = self.clip_n_frames
        seq_length = self.clip_n_frames #seq_length=hidden_size
        video_batch_size = int(ImageBatchSize/seq_length)

        self.LSTM_CNN = CNN_LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, seq_length=seq_length, video_batch_size=video_batch_size, num_classes=50)
        self.LSTM_CNN.load_state_dict(torch.load('/Users/elahehbaharlouei/Desktop/DL_Project/Code/Saved_Model/ResNet101.pth', map_location=torch.device('cpu')))
        self.LSTM_CNN.eval()
        print("Loaded model from disk")

        model_is_loaded = True
        count = 0
        count_frame = 0
        count_human = 0
        human_freq_rate = 0.6
        frames = torch.zeros((self.clip_n_frames, 3, 224, 224))
        time_to_appear = 5
        count_text = time_to_appear + 1

        while(self.running):
            if not flag_sync:
                start_time = time.time()
                # exist_human = self.processFrame(new_image)
                exist_human = self.processFrame()
                end_time = time.time()
                human_det_delay = end_time - start_time

                
                if exist_human:
                    count_human += 1
                
                if preprocess_enabled==1:
                    image_tmp = Image.fromarray(new_image)
                    frames[count_frame] = image_transform(image_tmp)
                else:
                    image_tmp = Image.fromarray(new_image)
                    frames[count_frame] = image_transform2(image_tmp)
                        
                
                # cv2.imshow('Camera', new_image)
                count = 0
                count_frame += 1
                
                if count_text > time_to_appear:
                    Text_on_Image = ""
                    Predict_Score = ""

                if (count_frame == self.clip_n_frames and count_human >= int(human_freq_rate*self.clip_n_frames)):
                    print("Processing")
                    flag_load_next = False
                    count_human = 0
                    count_frame = 0
                    count_text = 0

                    predicted_values = self.LSTM_CNN(frames)
                    # print("predicted_values:", predicted_values)
                    predicted_values_ = predicted_values.softmax(dim = 1)

                    max_val, max_id = torch.max(predicted_values_, dim=1)
                    print(f'max_id:{max_id}, max_val:{max_val}')
                    self.text_on_img(max_id, max_val)
                    
                    FPS_HUM_RATIO = human_det_delay
                    FPS_HUM_RATIO = (1/self.video_rate)/FPS_HUM_RATIO
                    # FPS_HUM_RATIO = self.video_rate
                    self.frame_rate = max(int(FPS_HUM_RATIO), 1)
                    print("Delay_1: ", human_det_delay, " -> ", "Frame Rate: ", self.frame_rate)

                    flag_load_next = True
                    
                    print("Done!")

                    
                elif (count_frame == self.clip_n_frames):
                    count_human = 0
                    count_frame = 0
                    
                    ### Updating the self.frame_rate
                    
                    
                    FPS_HUM_RATIO = human_det_delay
                    FPS_HUM_RATIO = (1/self.video_rate)/FPS_HUM_RATIO
                    # FPS_HUM_RATIO = self.video_rate
                    self.frame_rate = max(int(FPS_HUM_RATIO), 1)
                    
                    print("Delay_2: ", human_det_delay, " -> ", "Frame Rate: ", self.frame_rate)
                    
                if count_text <= time_to_appear:
                    count_text += 1

                flag_sync = True
                flag_load_next = True

    def text_on_img(self, max_id, max_val):
        global Text_on_Image,Predict_Score
        if max_id == 0:
            Text_on_Image = "Boxing:"
        elif max_id == 1:
            Text_on_Image = "Clapping:"
        elif max_id == 2:
            Text_on_Image = "Eating:"
        elif max_id == 3:
            Text_on_Image = "Hugging:"
        elif max_id == 4:
            Text_on_Image = "Jumping:"
        elif max_id == 5:
            Text_on_Image = "Laughing:"
        elif max_id == 6:
            Text_on_Image = "Smoking:"
        elif max_id == 7:
            Text_on_Image = "Walking:"
        elif max_id == 8:
            Text_on_Image = "HandWaiving:"
        Predict_Score = str(round(max_val[0].item(), 2))
    # def load_model_weights(self, network_model):

    #     # load json and create model
    #     # Network Model: ResNet101, Xception, Inception, VGG_16, etc.
    #     Model_json_file = open('./Saved_Model/' + network_model + '.json', 'r')
    #     loaded_model_json = Model_json_file.read()
    #     Model_json_file.close()
    #     self.loaded_Model = model_from_json(loaded_model_json)
    #     # load weights into new model
    #     self.loaded_Model.load_weights('./Saved_Model/Weights/' + network_model + '.h5')
    #     print("Loaded model from disk")
        
        
    #     LSTM_json_file = open('./Saved_Model/LSTM_Model_' + network_model + '.json', 'r')
    #     loaded_LSTM_json = LSTM_json_file.read()
    #     LSTM_json_file.close()
    #     self.loaded_LSTM = model_from_json(loaded_LSTM_json)
    #     # load weights into new model
    #     self.loaded_LSTM.load_weights('./Saved_Model/Weights/LSTM_Model_' + network_model + '.h5')
    #     print("Loaded model from disk")



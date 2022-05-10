# Human-Activity-Recognition
This project includes three main steps that I will explain separately here.

## The first step is preprocessing. You can find the .py and .ipynb codes of this step in Preprocessing folder.
In this step, we read the video dataset files and break them into some parts that each part has the same length. Then, I extract the fixed number of frames from each part, attach them, and save them as one big picture.
  ```
  ...
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
 ...
```
The reason is that the dataloader will read these images and shuffle them during the training step. But since all frames of one video are stored in one big picture, we will have them in the correct order in one picture so that we can separate them after reading from the dataset and before loading to the CNN-LSTM model. I repeat this process for all videos in different categories (9 categories - train and test) of my dataset and will store them in the related category in the final dataset. Note that the uploaded .py and .ipynb codes are only one version of my code, and I have three different versions based on the number of frames that we want to extract from each video **(20, 30, and 50 frames)**.

## The second step is training. You can find the .py and .ipynb codes in the Training folder.
In this step, I defined my **CNN_LSTM model** with details provided in the project report.
```
...
class CNN_LSTM(nn.Module):
    #We should have batch_size*seq_length batches everytime. So, the input is (batch_size*seq_length, C=3, H, W). 
    #After extracting features from model_CNN the output shape is (batch_size*seq_length, C_new, H_new, W_new).
    #We reshape the output to (batch_size, seq_length, -1) for passing to model_RNN
    def __init__(self, input_size=2048*1*1, hidden_size=4, num_layers=2, seq_length=20, video_batch_size=16, num_classes=50):
	self.seq_length = seq_length
	self.video_batch_size = video_batch_size
	super(CNN_LSTM, self).__init__()
	#
	self.model_CNN = models.resnext101_32x8d(pretrained=True)
	for param in self.model_CNN.parameters():
	    param.requires_grad = False
	#
	# x -> (batch_size, seq_length, input_size) --> seq_length=hidden_size
	# nn.LSTM(input_size, hidden_size, num_layers) --> hidden_size=seq_length, num_layers=how many LSTM layer we want to stack together
	self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
	#
	self.h0 = torch.zeros(num_layers*2, video_batch_size, hidden_size).to(device=device)
	self.c0 = torch.zeros(num_layers*2, video_batch_size, hidden_size).to(device=device)
	#
	self.fc1 = nn.Linear(hidden_size*2, 512)
	self.drp1 = nn.Dropout(0.5)
	kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
	constant_(self.fc1.bias, 0)
	self.act1 = nn.ReLU()
	self.fc2 = nn.Linear(512, 256)
	self.drp2 = nn.Dropout(0.4)
	kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
	constant_(self.fc2.bias, 0)
	self.act2 = nn.ReLU()
	#
	self.fc3 = nn.Linear(256, num_classes)
	kaiming_uniform_(self.fc3.weight)
	constant_(self.fc2.bias, 0)
	#
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
	#
	x = self.drp1(self.act1(self.fc1(out_lstm[:, -1, :])))
	x = self.drp2(self.act2(self.fc2(x)))
	x = self.fc3(x)
	#
	return(x)
	# return(out_lstm)
	#
    def get_activation(self, name):
	def hook(model, input, output):
	    self.CNN_Feature_Out[name] = output.detach()
	return hook
...
```
Then, I created the train and test dataloader, instantiated the model, and defined the optimizer and train_model function to start training and testing the model. Different parameters considered in this step are based on preprocessing step, especially the _seq_length_ of the LSTM model. I find the ideal amount for other hyperparameters by experience and based on the results. You can find the best parameters that I've got in the mentioned files. Note that the uploaded .py and .ipynb codes are only one version of my code, and I have **24** different versions based on the **basic CNN model (ResNet50 and ResNet101)**, **seq_length (20, 30, and 50 frames)**, and the **CNN-LSTM model complexity (2 versions: simple and more complicated)**. 

## The third step is GUI. I will break this step into three different parts to explain it easier. You can find the related files in the GUI folder. Also, there are two folders, Person_Detection and Save_Model, which contain the pre-trained object detection model and pre-trained CNN-LSTM model, respectively. 
### The first part is designing the GUI. 
I used tkinter to design my GUI. Different actions have been defined for different modes in GUI. We have two modes: Real-Time and Offline detection. Based on the user selection, we call the related class.
```
...
class ActivityDetector:
    ...
    def on_button_manual(self):
        ...
    def on_button_real(self):
        ...
    def Initialize_Objects(self):
        ...
    def sel_real(self):
        ...
    def sel_manual(self):
        ...
    def change_dropdown(self, *args):
        ...
    def change_dropdown_time(self, *args):
        ...
    def change_dropdown_real(self, *args):
        ...
    def center_window(self, width=300, height=200):
        ...
    def close_windows(self):
        ...
    def sel_chk_real(self):
        ...
    def sel_chk_manual(self):
        ...
    def sel_browse_video(self):
        ...
...
```
### The second part is the real-time detection class. 
This class has several functions. The first two functions are related to the human detection model. I used the existed code for loading and running the pre-trained mobilenet object detection model. These two functions are:
```
...
def __init__(self, path_to_ckpt):
    ...
def processFrame(self):    
    ...
...
```
The following three functions do the main process in real-time detection, which are my main contribution to this part.
```
...
def real_time_load(self, network_model, preprocessing):
    ...
def prediction(self):
    ...
def text_on_img(self, max_id, max_val):
    ...
...
```
The first function gets the stream of frames from the camera based on a specific frame rate and passes them to the second function. I used general variables to do this passing. In the prediction function, I get the current frame and call the human detection model to detect human existence and count the number of frames that humans existed on them. Then, I check this counter with a threshold value. If it's more than the threshold, I call the trained CNN-LSTM model and do the prediction on these frames. Next, based on the prediction, I call the text_on_img function to extract the name of the predicted category and the amount of prediction. I will show these numbers on the last frames for a specific time. 
### The third part is the Offline detection class. 
This class is similar to the real-time detection model and almost does the same process, except instead of reading frames from the camera, it extracts them from loaded video file.
```
...
def manual_prediction(self, network_model, preprocessing):
    ...
def prediction(self):
    ...
def text_on_img(self, max_id, max_val):
    ...
...
```



from Real_Time_test import preprocess
from Manual_test import preprocess_manual

import tkinter as tk
from tkinter import filedialog

class ConfusionDetector:
    def __init__(self, master):
        self.master = master
        self.Initialize_Objects()
        

    def on_button_manual(self):
        str_model = self.tkvar.get()
        if str_model == "ResNet101":
            self.model_name = "ResNet101"
        else:
            self.model_name = self.tkvar.get()
            
        path_test = self.entry1.get()
        self.model_name = self.tkvar.get()
        sec_per_batch = int(self.tkvar_time.get())
        
        print(path_test)
        print(self.model_name)
        print(sec_per_batch)
        
        model_path = './Person_Detection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
            
        video_path = path_test
        clip_time = sec_per_batch
        sample = preprocess_manual(model_path, clip_time, video_path)
        
        # sample.load_model_weights(self.model_name)
        # sample.manual_prediction(self.var_chk_man.get())
        sample.manual_prediction(self.model_name, self.var_chk_man.get())

        # self.newWindow = tk.Toplevel(self.master)
        # self.app = self.Call_Predict(self.newWindow)
        
        
    def on_button_real(self):
        str_model = self.tkvar_real.get()
        if str_model == "ResNet101":
            self.model_name = "ResNet101"
        else:
            self.model_name = self.tkvar_real.get()
            
        print(self.model_name)

        model_path = './Person_Detection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
            
        sample = preprocess(model_path)
        
        # sample.load_model_weights(self.model_name)
        
        sample.real_time_load(self.model_name, self.var_chk_real.get())
        
        # self.newWindow = tk.Toplevel(self.master)
        # self.app = self.Call_Predict(self.newWindow)
        
        
    def Initialize_Objects(self):
        self.frame = tk.Frame(self.master)
        
        self.center_window(750, 300)
        
        #tk.Tk.title("Confusion Detection")
        object_length = 40
        showfont="Georgia 16 bold" # define a consistent font to use
        radio_color_bg = 'gray'
        radio_color_fg = 'white'
        label_color_bg = 'light gray'
        label_color_fg = 'black'
        
        count_row = 0
        count_col = 0
        
        # Create a Tkinter variable
        self.tkvar_real = tk.StringVar(self.master)
        self.tkvar = tk.StringVar(self.master)
        self.tkvar_time = tk.StringVar(self.master)
        
        # Dictionary with options
        choices_real = { 'VGG19','VGG16','ResNet101'}
        self.tkvar_real.set('ResNet101') # set the default option
        # link function to change dropdown
        self.tkvar_real.trace('w', self.change_dropdown_real)
        
        choices = { 'VGG19','VGG16','ResNet101'}
        self.tkvar.set('ResNet101') # set the default option
        # link function to change dropdown
        self.tkvar.trace('w', self.change_dropdown)
        
        choices_time = { '2','3','4', '5'}
        self.tkvar_time.set('4') # set the default option
        # link function to change dropdown
        self.tkvar_time.trace('w', self.change_dropdown_time)
        
        count_row += 1
        self.var = tk.IntVar()  # all Radiobutton widgets will be set to the same control variable

        self.R1 = tk.Radiobutton(self.master, anchor='w', text="Real-Time Prediction", variable=self.var, value=1, command=self.sel_real, 
                                 font=showfont, bg=radio_color_bg, fg=radio_color_fg, relief="raised")
        self.R1.select()
        self.R1.grid(row=count_row, column=count_col, sticky=tk.W)
        
        count_col += 1
        self.var_chk_real = tk.IntVar()
        self.chkpre_real = tk.Checkbutton(self.master, text="Preprocess Input", variable=self.var_chk_real, command=self.sel_chk_real, state=tk.NORMAL)
        self.chkpre_real.grid(row=count_row, column=count_col, sticky=tk.W)
        self.chkpre_real.select()
        ##################################
        #######Real Time Prediction#######
        ##################################
        count_row += 1
        count_col = 0
        self.lbl0 = tk.Label(self.master, text='Please Enter Following Entries before Running:', anchor='w', bg=radio_color_bg, fg=radio_color_fg, width=object_length, relief="raised") 
        self.lbl0.grid(row=count_row, column=count_col, sticky=tk.W)
        
        count_row += 1
        count_col = 0
        #model name
        self.lbl_Rl = tk.Label(self.master, anchor='w', text='Model Name (ResNet101, VGG16, VGG19):', bg=label_color_bg, fg=label_color_fg, width=object_length, relief="raised")
        self.lbl_Rl.grid(row=count_row, column=count_col, sticky=tk.W)
        count_col += 1
        
        self.popupMenu_Rl = tk.OptionMenu(self.master, self.tkvar_real, *choices_real)
        self.popupMenu_Rl.config(width=object_length-2, state=tk.NORMAL)
        self.popupMenu_Rl.grid(row = count_row, column =count_col, sticky=tk.W)
        
        # self.entry2_Rl = tk.Entry(self.master, textvariable=v2_Rl, width=object_length)
        # self.entry2_Rl.grid(row=count_row, column=count_col, sticky=tk.E)
        
        count_row += 1
        self.button_Rl = tk.Button(self.master, text='Start Prediction!', state=tk.NORMAL, command=self.on_button_real, width=int(object_length/2))
        self.button_Rl.grid(row=count_row, column=count_col, sticky=tk.W)
        
        self.button_Rl2 = tk.Button(self.master, text='Exit', state=tk.NORMAL, command=self.close_windows, width=int(object_length/2))
        self.button_Rl2.grid(row=count_row, column=count_col, sticky=tk.E)
        
        
        ###############################
        #######Manual Prediciton#######
        ###############################
        count_row += 1
        count_col = 0
        self.R1 = tk.Radiobutton(self.master, anchor='w', text="Offline Prediction", variable=self.var, value=2, command=self.sel_manual, 
                                 font=showfont, bg=radio_color_bg, fg=radio_color_fg, relief="raised")
        self.R1.grid(row=count_row, column=count_col, sticky=tk.W)
        
        count_col += 1
        self.var_chk_man = tk.IntVar()
        self.chkpre_manual = tk.Checkbutton(self.master, text="Preprocess Input", variable=self.var_chk_man, command=self.sel_chk_manual, state=tk.DISABLED)
        self.chkpre_manual.grid(row=count_row, column=count_col, sticky=tk.W)
        self.chkpre_manual.select()
        
        count_row += 1
        count_col = 0
        self.lbl0 = tk.Label(self.master, text='Please Enter Following Entries before Running:', anchor='w', bg=radio_color_bg, fg=radio_color_fg, width=object_length, relief="raised") 
        self.lbl0.grid(row=count_row, column=count_col, sticky=tk.W)
        
        count_row += 1
        count_col = 0
        object_length = 40
        #video path
        self.lbl1 = tk.Label(self.master, text='Video Path:', anchor='w', bg=label_color_bg, fg=label_color_fg, width=int(object_length/2), relief="raised") 
        self.lbl1.grid(row=count_row, column=count_col, sticky=tk.W)
        
        self.button_browse = tk.Button(self.master, text='Browse', state=tk.DISABLED, command=self.sel_browse_video, width=int(object_length/2))
        self.button_browse.grid(row=count_row, column=count_col, sticky=tk.E)
        
        count_col += 1
        self.entry1_text = tk.StringVar()
        self.entry1 = tk.Entry(self.master, textvariable=self.entry1_text, width=object_length, state=tk.DISABLED)
        self.entry1.grid(row=count_row, column=count_col, sticky=tk.E)
        
        count_row += 1
        count_col = 0
        #model name
        self.lbl2 = tk.Label(self.master, anchor='w', text='Model Name (InceptionResNetV2, InceptionV3):', bg=label_color_bg, fg=label_color_fg, width=object_length, relief="raised")
        self.lbl2.grid(row=count_row, column=count_col, sticky=tk.W)
        v2 = tk.StringVar()
        
        count_col += 1
        self.popupMenu = tk.OptionMenu(self.master, self.tkvar, *choices)
        self.popupMenu.config(width=object_length-2, state=tk.DISABLED)
        self.popupMenu.grid(row = count_row, column =count_col, sticky=tk.W)
        
        # self.entry2 = tk.Entry(self.master, textvariable=v2, width=object_length)
        # self.entry2.grid(row=count_row, column=count_col, sticky=tk.E)
        
        count_row += 1
        count_col = 0
        #seconds per batch
        self.lbl3 = tk.Label(self.master, anchor='w', text='Seconds per batch:', bg=label_color_bg, fg=label_color_fg, width=object_length, relief="raised")
        self.lbl3.grid(row=count_row, column=count_col, sticky=tk.W)
        v3 = tk.StringVar()
        count_col += 1
        self.popupMenu_time = tk.OptionMenu(self.master, self.tkvar_time, *choices_time)
        self.popupMenu_time.config(width=object_length-2, state=tk.DISABLED)
        self.popupMenu_time.grid(row = count_row, column =count_col, sticky=tk.W)
        
        #self.entry3 = tk.Entry(self.master, textvariable=v3, width=object_length, state=tk.DISABLED)
        #self.entry3.grid(row=count_row, column=count_col, sticky=tk.E)
        
        count_row += 1
        self.button = tk.Button(self.master, text='Start Prediction!', state=tk.DISABLED, command=self.on_button_manual, width=int(object_length/2))
        self.button.grid(row=count_row, column=count_col, sticky=tk.W)
        
        self.button2 = tk.Button(self.master, text='Exit', state=tk.DISABLED, command=self.close_windows, width=int(object_length/2))
        self.button2.grid(row=count_row, column=count_col, sticky=tk.E)
        
        
    def sel_real(self):
        selection = "You selected the option  " + str(self.var.get())
        print(selection)
        self.button_Rl.config(state = tk.NORMAL)
        self.button_Rl2.config(state = tk.NORMAL)
        self.button.config(state = tk.DISABLED)
        self.button2.config(state = tk.DISABLED)
        self.chkpre_manual.config(state = tk.DISABLED)
        self.chkpre_real.config(state = tk.NORMAL)
        self.button_browse.config(state = tk.DISABLED)
        
        self.popupMenu.config(state=tk.DISABLED)
        self.popupMenu_time.config(state=tk.DISABLED)
        self.entry1.config(state=tk.DISABLED)
        self.popupMenu_Rl.config(state=tk.NORMAL)
        
     
    def sel_manual(self):
        selection = "You selected the option  " + str(self.var.get())
        print(selection)
        self.button_Rl.config(state = tk.DISABLED)
        self.button_Rl2.config(state = tk.DISABLED)
        self.button.config(state = tk.NORMAL)
        self.button2.config(state = tk.NORMAL)
        self.chkpre_manual.config(state = tk.NORMAL)
        self.chkpre_real.config(state = tk.DISABLED)
        self.button_browse.config(state = tk.NORMAL)
        
        self.popupMenu.config(state=tk.NORMAL)
        self.popupMenu_time.config(state=tk.NORMAL)
        self.entry1.config(state=tk.NORMAL)
        self.popupMenu_Rl.config(state=tk.DISABLED)
        
    
    # on change dropdown value
    def change_dropdown(self, *args):
        print( self.tkvar.get() )
            
        
    # on change dropdown value
    def change_dropdown_time(self, *args):
            print( self.tkvar_time.get() )
        
    # on change dropdown value
    def change_dropdown_real(self, *args):
            print( self.tkvar_real.get() )
        
    def center_window(self, width=300, height=200):
        # get screen width and height
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
    
        # calculate position x and y coordinates
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.master.geometry('%dx%d+%d+%d' % (width, height, x, y))
        
    def close_windows(self):
        self.master.destroy()
    
    def sel_chk_real(self):
        print(self.var_chk_real.get())
    
    def sel_chk_manual(self):
        print(self.var_chk_man.get())
        
    def sel_browse_video(self):
        rep = filedialog.askopenfilenames(
    	parent=self.master,
    	initialdir='/',
        title = "Select file",
    	initialfile='./',
        defaultextension='.mov',
    	filetypes=[
    		("MOV", "*.mov"),
    		("mp4", "*.mp4"),
    		("AVI", "*.avi"),
    		("GIF", "*.gif"),
    		("All files", "*")])
        
        if rep != "":
            print(rep[0])
            self.entry1_text.set(rep[0])
   
def main(): 
    root = tk.Tk()
    app = ConfusionDetector(root)
    root.mainloop()

if __name__ == '__main__':
    main()    
        
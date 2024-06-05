from tkinter import*    #GUI
from tkinter import ttk  #GUI
from PIL import Image,ImageTk    #pip install pillow for image
from tkinter import messagebox


def call_GUI1():    #for Static Button
    win2 = Toplevel(root)
    Second_Window(win2)
    return

def call_GUI2():    #For Real Time Button
    win3 = Toplevel(root)
    Third_Window(win3)
    win3.destroy()
    return

#Main window
class First_Window:     
    def __init__(self,root):    
        self.root=root
        self.root.title("Main")

        screen_width=root.winfo_screenwidth()   #Fetching screen width
        screen_height=root.winfo_screenheight()     #Fetching screen height
        root.geometry(f'{screen_width}x{screen_height-100}')    #Geometry For main  window and -100 so that it will not loss any part of main window

        img1 = Image.open("images/2-AI-invades-automobile-industry-in-2019.jpeg") #AI Hand Image
        img1 = img1.resize((1530,800), Image.ANTIALIAS)
        self.photoImg1 = ImageTk.PhotoImage(img1)
        bg_lbl=Label(self.root,image=self.photoImg1)
        bg_lbl.place(x=0,y=0,width=1530,height=800)

        title=Label(bg_lbl,text="Emotion Detection using AI",font=("times new roman",35,"bold"),bg="white",fg="red") #WHite strip of main
        title.place(x=0,y=120,width=1550,height=45)

        myname=Label(self.root,text="Developed By:A Square P",fg="black",bg="white",font=("times new roman",18,"bold"))#Developed by
        myname.place(x=0,y=0)
        
        img10 = Image.open("images/facial-recognition_0.jpg")   #image displaying of facial recognization
        img10 = img10.resize((500,120), Image.ANTIALIAS)
        self.photoImg10 = ImageTk.PhotoImage(img10)
        bg_lbl1=Label(bg_lbl,image=self.photoImg10)
        bg_lbl1.place(x=0,y=0,width=500,height=120)

        img11 = Image.open("images/facialrecognition.png")
        img11 = img11.resize((500,120), Image.ANTIALIAS)
        self.photoImg11 = ImageTk.PhotoImage(img11)
        bg_lbl22=Label(bg_lbl,image=self.photoImg11)
        bg_lbl22.place(x=500,y=0,width=500,height=120)

        img13 = Image.open("images/smart-attendance.jpg")
        img13 = img13.resize((550,120), Image.ANTIALIAS)
        self.photoImg13= ImageTk.PhotoImage(img13)
        bg_lbl12=Label(bg_lbl,image=self.photoImg13)
        bg_lbl12.place(x=1000,y=0,width=550,height=120)


        frame=Frame(self.root,bg="black")
        frame.place(x=610,y=200,width=340,height=430)

        img1=Image.open("images/LoginIconAppl.png")
        img1=img1.resize((90,90),Image.ANTIALIAS)
        self.photoimage1=ImageTk.PhotoImage(img1)
        lblimg1=Label(image=self.photoimage1,bg="black",borderwidth=0)
        lblimg1.place(x=730,y=200,width=90,height=90)

        get_str=Label(frame,text="Get Started",font=("times new roman",20,"bold"),fg="white",bg="black")
        get_str.place(x=95,y=85)  

        # LoginButton
        btn_login=Button(frame,text="STATIC",borderwidth=5,relief=RAISED,command=call_GUI1,cursor="hand2",font=("times new roman",20,"bold"),fg="white",bg="red" ,activebackground="#B00857")
        btn_login.place(x=75,y=160,width=200,height=50)

        btn_login1=Button(frame,text="REAL TIME",borderwidth=5,relief=RAISED,command=call_GUI2,cursor="hand2",font=("times new roman",20,"bold"),fg="white",bg="red" ,activebackground="#B00857")
        btn_login1.place(x=75,y=270,width=200,height=50)

    '''def return_login(self):
        self.root.destroy()
       '''     

import tkinter as tk
class Second_Window:     
    def __init__(self,root):    
        self.root=root
        self.root.title("Static")
        screen_width=root.winfo_screenwidth()   #Fetching screen width
        screen_height=root.winfo_screenheight()     #Fetching screen height
        root.geometry(f'{screen_width}x{screen_height-100}')


        frame=Frame(self.root,bg="black")
        frame.place(x=610,y=200,width=340,height=430)

        #entry1 = tk.Entry (root) 
        #frame.create_window(200, 140, window=entry1)
        self.var_SecurityA=StringVar()
        entry1=Entry(frame,textvariable= self.var_SecurityA,bd=5,relief=GROOVE,width=20,font=("times new roman",18))
        entry1.grid(row=7,column=1,padx=20,pady=3)
        def getLink ():  
            x1 = entry1.get()
            import cv2
            import numpy as np
            from keras.models import model_from_json


            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

            # load json and create model
            json_file = open('model/emotion_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            emotion_model = model_from_json(loaded_model_json)

            # load weights into new model
            emotion_model.load_weights("model/emotion_model.h5")
            print("Loaded model from disk")

            # start the webcam feed
            #cap = cv2.VideoCapture(0)

            # pass here your video path
            # you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
            cap = cv2.VideoCapture(x1)

            while True:
                # Find haar cascade to draw bounding box around face
                ret, frame = cap.read()
                frame = cv2.resize(frame, (1280, 720))
                if not ret:
                    break
                face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces available on camera
                num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                # take each face available on the camera and Preprocess it
                for (x, y, w, h) in num_faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                    # predict the emotions
                    emotion_prediction = emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Emotion Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

            

        button1=Button(frame,text="Get Link",borderwidth=5,relief=RAISED,command=getLink,cursor="hand2",font=("times new roman",20,"bold"),fg="white",bg="red" ,activebackground="#B00857")
        button1.place(x=75,y=160,width=200,height=50)

        button2=Button(frame,text="QUIT",borderwidth=5,relief=RAISED,command=root.destroy,cursor="hand2",font=("times new roman",20,"bold"),fg="white",bg="red" ,activebackground="#B00857")
        button2.place(x=75,y=270,width=200,height=50)

        '''button1 = tk.Button(text='Get the Link', command=getSquareRoot)
        canvas1.create_window(200, 180, window=button1)

        button2 = tk.Button(text='Quit', command=root.destroy)
        canvas1.create_window(200, 200, window=button2)'''


from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

class Third_Window:     
    def __init__(self,root):    
        self.root=root
        self.root.title("Real-Time")

        screen_width=root.winfo_screenwidth()   #Fetching screen width
        screen_height=root.winfo_screenheight()     #Fetching screen height
        root.geometry(f'{screen_width}x{screen_height-100}')

        face_classifier = cv2.CascadeClassifier(
            r'haarcascade_frontalface_default.xml')
        classifier = load_model(
            r'model.h5')

        emotion_labels = ['Angry', 'Disgust', 'Fear',
                        'Happy', 'Neutral', 'Sad', 'Surprise']

        cap = cv2.VideoCapture(0)


        while True:
            _, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    root=Tk()
    app=First_Window(root)
    root.mainloop()
    

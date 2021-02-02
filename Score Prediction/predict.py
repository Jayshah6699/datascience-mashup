from tkinter import *
from PIL import Image, ImageTk
import numpy as np           
import pandas as pd
import pickle


class Track:
        
    def __init__(self):
        # Defining the root window
        self.root = Tk()
        self.root.geometry("2000x1024")
        self.root.title("Score Predictor !!")
                
        # Defining the canvas on which the widgets has to be placed
        self.c = Canvas(self.root,bg = "gray",height=2000,width=2024,cursor='pencil')
        
        # Setting the background image and scaling it
        image = Image.open("images/test.jpg")
        image = image.resize((1550, 820), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image)
        
        # Setting the background
        self.c.create_image((0,0), image=photo, anchor="nw")
        
        
        # Setting the font
        self.fnt = ('latin modern typewriter',20,'bold')
        
        # Setting the text
        self.c.create_text((200, 200),text="Predict your score now !!", fill="red", anchor="nw"
                           ,font=('newcenturyschlbk',30,'bold'))
        
        
        ## note text
        
        self.c.create_text((900,700),text = "Note: Maximum marks is kept for 100", fill = "black",anchor = "nw",font=("newcenturyschlbk",20,'bold'))
        
    
        self.c.create_text((200, 300), text="Enter number of study hour ", fill="blue", anchor="nw"
                           ,font=self.fnt)
        
        # Setting the country entry boxes
        self.c.create_text((200, 350), text="Hour(s): ", fill="black", anchor="nw"
                           ,font=('Times',20,'italic bold'))
        
        self.button1 = Entry(self.c,font=('Times',20,'bold'))
        self.button1.configure(width = 10, relief = FLAT)  
        button1_window = self.c.create_window(300, 350, anchor=NW, window=self.button1)
        
        
        
        # predict button
        self.enterButton = Button(self.c,text="Predict",width=15,height=2,bg='blue',fg='white',command = self.show_data,
                                  font=("Times",10,'bold'))
        self.enterButton.configure(width=10,relief=FLAT)
        login_button_window = self.c.create_window(300,400,anchor=NW,window=self.enterButton)
        
        
        
        
        ## clear button
        self.enterButton = Button(self.c,text="Clear",width=15,height=2,bg='red',fg='white',command = self.clear,
                                  font=("Times",10,'bold'))
        self.enterButton.configure(width=10,relief=FLAT)
        login_button_window = self.c.create_window(400,400,anchor=NW,window=self.enterButton)
        
        
        ## marks entry box
        self.c.create_text((200, 450), text="Marks Obtained: ", fill="black", anchor="nw"
                           ,font=('Times',20,'italic bold'))
        
        self.button2 = Entry(self.c,font=('Times',20,'bold'))
        self.button2.configure(width = 10, relief = FLAT)  
        button2_window = self.c.create_window(400, 450, anchor=NW, window=self.button2)
        
        
        
        
        
        # Defining the window
        self.c.pack()
        self.root.mainloop()
        
        
    def show_data(self):
        filename = 'student_score.h5'
        model = pickle.load(open(filename, 'rb'))
        hour = self.button1.get()
        hour  = float(hour)
        print(hour)
        
        if hour <=9.9:
            
            res =  int(round(model.predict([[hour]])[0]))
            print(int(res))
            
            self.button2.insert(0,int(res))
        
        else:
            
            self.button2.insert(0,100)
            
            
        if hour >=10:
             # Setting the text
            self.text_id =  self.c.create_text((800, 200),text="You will ace the exam, take a chill pill !!", fill="blue", anchor="nw"
                           ,font=('newcenturyschlbk',20,'bold'))
        elif hour < 10 and hour >=9:
            self.text_id =  self.c.create_text((800, 200),text="Excellent, you are working hard !!", fill="blue", anchor="nw"
                           ,font=('newcenturyschlbk',20,'bold'))
        elif hour < 9 and hour >=6:
            self.text_id =  self.c.create_text((800, 200),text="Good, spend little more time on studies!", fill="blue", anchor="nw"
                           ,font=('newcenturyschlbk',20,'bold'))
        elif hour < 6 and hour >=4:
            self.text_id =  self.c.create_text((800, 200),text=" Invest time on studies !!", fill="blue", anchor="nw"
                           ,font=('newcenturyschlbk',20,'bold'))
        else:
            self.text_id =  self.c.create_text((800, 200),text="Seriously , work hard !!", fill="blue", anchor="nw"
                           ,font=('newcenturyschlbk',20,'bold'))
            
            
        
        
        
        
    
    def clear(self):
        
        self.button1.delete(0,'end')
        self.button2.delete(0,'end')
        self.c.delete(self.text_id)

track = Track()
        
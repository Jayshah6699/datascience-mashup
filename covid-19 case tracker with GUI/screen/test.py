from tkinter import *
from PIL import Image, ImageTk
import covid                               
import matplotlib.pyplot as plt            
import pandas as pd     




class Track:
        
    def __init__(self):
        # Defining the root window
        self.root = Tk()
        self.root.geometry("2000x1024")
        self.root.title("COVID- Live Status Tracker")
                
        # Defining the canvas on which the widgets has to be placed
        self.c = Canvas(self.root,bg = "gray",height=2000,width=2024,cursor='heart')
        
        # Setting the background image and scaling it
        image = Image.open("images/corona19.jpg")
        image = image.resize((1600, 820), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image)
        
        # Setting the background
        self.c.create_image((0,0), image=photo, anchor="nw")
        
        
        # Setting the font
        self.fnt = ('latin modern typewriter',30,'bold')
        
        # Setting the text
        self.c.create_text((300, 200),text="COVID -19 Live Status !!", fill="white", anchor="nw"
                           ,font=('newcenturyschlbk',30,'bold'))
        
    
        self.c.create_text((200, 300), text="Enter your country name ", fill="light yellow", anchor="nw"
                           ,font=self.fnt)
        
        # Setting the country entry boxes
        self.c.create_text((300, 400), text="Country: ", fill="white", anchor="nw"
                           ,font=('Times',20,'italic bold'))
        
        self.button1 = Entry(self.c,font=('Times',20,'bold'))
        self.button1.configure(width = 20, relief = FLAT)  
        button1_window = self.c.create_window(500, 400, anchor=NW, window=self.button1)
        
        self.c.create_text((300, 450), text="Active cases: ", fill="white", anchor="nw"
                           ,font=('Times',20,'italic bold'))
        
        
        self.button2 = Entry(self.c,font=('Times',20,'bold'))
        self.button2.configure(width = 20, relief = FLAT)  
        button2_window = self.c.create_window(500, 450, anchor=NW, window=self.button2)
        
        
        self.c.create_text((300, 500), text="Death cases: ", fill="white", anchor="nw"
                           ,font=('Times',20,'italic bold'))
        
        self.button3 = Entry(self.c,font=('Times',20,'bold'))
        self.button3.configure(width = 20, relief = FLAT)  
        button3_window = self.c.create_window(500, 500, anchor=NW, window=self.button3)
        
        
        self.c.create_text((300, 550), text="Confirmed cases: ", fill="white", anchor="nw"
                           ,font=('Times',20,'italic bold'))
        
        self.button4 = Entry(self.c,font=('Times',20,'bold'))
        self.button4.configure(width = 20, relief = FLAT)  
        button4_window = self.c.create_window(500, 550, anchor=NW, window=self.button4)
        
        
        self.c.create_text((300, 600), text="Recovered cases: ", fill="white", anchor="nw"
                           ,font=('Times',20,'italic bold'))
        
        
        self.button5 = Entry(self.c,font=('Times',20,'bold'))
        self.button5.configure(width = 20, relief = FLAT)  
        button5_window = self.c.create_window(500, 600, anchor=NW, window=self.button5)
        
        
        
        
        self.enterButton = Button(self.c,text="Click here",width=15,height=2,bg='blue',fg='white',command = self.show_data,
                                  font=("Times",10,'bold'))
        self.enterButton.configure(width=10,relief=FLAT)
        login_button_window = self.c.create_window(800,400,anchor=NW,window=self.enterButton)
        
        
        # Defining the window
        self.c.pack()
        self.root.mainloop()
        
    def show_data(self):
        
            
        data = covid.Covid()
        countryname = self.button1.get()
        print(countryname)
        
        status = data.get_status_by_country_name(countryname)
        active = status['active']
        self.button2.insert(0,active)
        death = status['deaths']
        self.button3.insert(0, death)
        confirm = status['confirmed']
        self.button4.insert(0, confirm)
        recover = status['recovered']
        self.button5.insert(0, recover)
        print(status)
            # intialise data of lists.
        data = {'id': status['id'],
                    'Country': status['country'],
                    'Confirmed': status['confirmed'],
                    'Active': status['active'],
                    'Deaths': status['deaths'],
                    'Recovered': status['recovered'],
                    'Latitude': status['latitude'],
                    'Longitude': status['longitude'],
                    'Last_Updated': status['last_update']
                    }

            # Create DataFrame
        df = pd.DataFrame(data, index=[0])

            # Print the output.
        print(df)
        cadr = {

                key:status[key]
                for key in status.keys() & {"confirmed","active","deaths","recovered"}
            }
        n = list(cadr.keys())
        v = list(cadr.values())
        plt.title("Country")
        plt.bar(range(len(cadr)),v,tick_label=n,label=('active'))
        plt.xlabel('x-labels')
        plt.ylabel('data')

        plt.plot(range(len(cadr)))


        plt.show()
    
    
    
        
        
        



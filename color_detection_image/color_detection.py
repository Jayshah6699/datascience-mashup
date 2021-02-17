# Color Detection Using OpenCV and Pandas

# importing libraries 
import cv2
import pandas as pd


img_path = 'lion.jfif'
csv_path = 'colors.csv'

# reading csv file
index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv(csv_path,names=index, header=None)
print(df.head(5))


#print(len(df))
img = cv2.imread(img_path)
#print(img)
img = cv2.resize(img , (800,600)) 


clicked =  False
r = g = b = xpos = ypos = 0


# To know the color name with the help of csv file data 
def get_color_name(R ,G ,B):
    minimum = 1000
    for i in range(len(df)):
        d = abs(R - int(df.loc[i,'R'])) + abs(G - int(df.loc[i,'G'])) + abs(B - int(df.loc[i,'B']))   
        if d  <= minimum:
           minimum = d
           cname = df.loc[i ,'color_name']
    
    return cname



# draw function to know the color pixels at a particular point in the image
def draw_function(event , x , y , flags , params):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global clicked , r , g ,b , xpos , ypos 
        clicked = True
        xpos = x
        ypos = y
        b ,g, r = img[y,x] 
        b = int(b)
        g = int(g)
        r = int(r)
        


# showing image in window
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_function)

# loop to make program running to give color name 
while True:
    cv2.imshow('image', img)
    if clicked:
        cv2.rectangle(img, (20,20), (600,60), (b,g,r) ,-1)
        text = get_color_name(r,g,b) + 'R=' + str(r) + 'G='+ str(g) + 'B='+ str(b)
        cv2.putText(img, text, (50,50), 2, 0.8,(0,0,0),2,cv2.LINE_AA)
    
    if cv2.waitKey(20) & 0xFF ==27:
        break
        
# one cannot close the image window by clicking close button at top
# either wait for 20 seconds or press esc key        
cv2.destroyAllWindows()



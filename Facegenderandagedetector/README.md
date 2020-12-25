# Face-gender-and-age-detector
This is a python script for face gender and age detector for an image and also for live webcam footage.
<p>In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. The identified faces may be predicted in gender either be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0-3), (4-7), (8-14), (15-20), (21-36), (38-46), (48-58), (60-100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.</p>

<h2>Dataset :</h2>
<p>For this python project, I had used the Adience dataset; the dataset is available in the public domain and you can find it <a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification">here</a>. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models I used had been trained on this dataset.</p>

<h2>Additional Python Libraries Required :</h2>
<ul>
  <li>OpenCV</li>
  
       pip install opencv-python
</ul>
<ul>
 <li>argparse</li>
  
       pip install argparse
</ul>

<h2>The contents of this Project :</h2>
<ul>
  <li>opencv_face_detector.pbtxt</li>
  <li>opencv_face_detector_uint8.pb</li>
  <li>age_deploy.prototxt</li>
  <li>age_net.caffemodel</li>
  <li>gender_deploy.prototxt</li>
  <li>gender_net.caffemodel</li>
  <li>a few pictures to try the project on</li>
  <li>FaceGenderandAgedetection.py</li>
 </ul>
 <p>For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.</p>
 
 <h2>Usage :</h2>
 <ul>
  <li>Download this repo</li>
  <li>Open your Command Prompt or Terminal and change directory to the folder where all the files are present.</li>
  <li><b>Detecting Gender and Age of face in Image</b> Use Command :</li>
  
      python FaceGenderandAgedetection.py --image <image_name>
</ul>
  <p><b>Note: </b>The Image should be present in same folder where all the files are present</p> 
<ul>
  <li><b>Detecting Gender and Age of face through webcam</b> Use Command :</li>
  
      python FaceGenderandAgedetection.py
</ul>
<ul>
  <li>Press <b>Ctrl + C</b> to stop the program execution.</li>
</ul>

# Working:

<h2>Examples :</h2>
<p><b>NOTE:- I downloaded the images from Google,if you have any query or problem i can remove them, i just used it for Educational purpose.</b></p>

    >python FaceGenderandAgedetection.py --image girl1.jpg
    Gender: Female
    Age: 15-20 years
    
<img src="Image Example/Face gender and age detection girl1.PNG">

    >python FaceGenderandAgedetection.py --image girl2.jpg
    Gender: Female
    Age: 4-7 years
    
<img src="Image Example/Face gender and age detection girl2.PNG">

    >python FaceGenderandAgedetection.py --image kid1.jpg
    Gender: Female
    Age: 4-7 years    
    
<img src="Image Example/Detecting age and gender kid1.png">

    >python FaceGenderandAgedetection.py --image kid2.jpg
    Gender: Male
    Age: 0-3 years  
    
<img src="Image Example/Detecting age and gender kid2.png">

    >python FaceGenderandAgedetection.py --image man1.jpg
    Gender: Male
    Age: 38-46 years
    
<img src="Image Example/Face gender and age detection man1.PNG">

    >python FaceGenderandAgedetection.py --image man2.jpg
    Gender: Male
    Age: 21-36 years
    
<img src="Image Example/Face gender and age detection man2.PNG">

    >python FaceGenderandAgedetection.py --image woman1.jpg
    Gender: Female
    Age: 21-36 years
    
<img src="Image Example/Face gender and age detection woman1.PNG">

    >python FaceGenderandAgedetection.py --image woman2.jpg
    Gender: Female
    Age: 21-36 years
    
<img src="Image Example/Face gender and age detection woman2.PNG">

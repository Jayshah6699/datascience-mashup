import streamlit as st
from tensorflow.keras.models import model_from_json
from numpy import asarray 
import numpy as np
import base64 
from PIL import Image


json_file = open("ResNetModel.json","r")
loaded_json_model = json_file.read()
json_file.close()  


model = model_from_json(loaded_json_model)
model.load_weights("ResNetModelWeights.h5")
labels = list("ABCDEF")
st.title("Finger Sign Classification")

st.markdown("""
<style>
body {
    color: #000;
    background-color:white;
    

    
</style>
    """, unsafe_allow_html=True)

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Upload image ", type=["png","jpg","jpeg"])

if st.button('Predict'):
    
    if uploaded_file is None:
        st.error("Please Upload Image !!")
    else:
        img = Image.open(uploaded_file)
        img = img.resize((50,50))
        img = asarray(img)
        
        print(img.shape)
        img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    
        pred = labels[np.argmax(model.predict(img))]
        st.image(img, use_column_width=True)
        st.header("The uploaded image indicates sign of alphabet "+ pred)

    


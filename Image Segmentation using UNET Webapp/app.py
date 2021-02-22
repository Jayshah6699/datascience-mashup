import streamlit as st
from tensorflow.keras.models import model_from_json
from numpy import asarray 
import numpy as np
import base64 
from PIL import Image
import tensorflow as tf

st.title("Image Segmentation using UNet")
model = tf.keras.models.load_model('model.h5')

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
        img = Image.open(uploaded_file,)
        img = img.resize((256,256),Image.ANTIALIAS)
        img = asarray(img)
        img = img[:,:,:3]
        input_array = tf.keras.preprocessing.image.img_to_array(img)
        input_array = np.array([input_array])  # Convert single image to a batch.
        predictions = model.predict(input_array)
        
        st.image(predictions, use_column_width=True)

    


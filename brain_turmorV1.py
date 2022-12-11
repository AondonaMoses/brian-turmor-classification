# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 21:15:30 2022

@author: lenovo
"""

import keras
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2


st.title("Brain Turmor Classification APP")


def prediction_model(img, model):
    # Load the model
    model = keras.models.load_model(model)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage,(150,150))
    img = img.reshape(1,150,150,3)
    prediction = model.predict(img)
    prediction = np.argmax(prediction,axis=1)[0]

    return prediction



def main():
    uploaded_file = st.file_uploader("Choose a brain MRI ...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=False)
        st.write("")
        st.write("Please Wait for your RESULT...")
        label = prediction_model(image, 'C:/Users/lenovo/Desktop/brain deployment/brain-tumor-mri-classificationModel.hdf5')
        if label == 0:
            st.write(" **TEST RESULT: Glioma Tumor**")
        elif label == 1:
            st.write("**TEST RESULT: No Tumor**")
        elif label == 2:
            st.write("**TEST RESULT: Meningioma Tumor**")
        else:
            st.write("**TEST RESULT: Pituitary Tumor**")
    else:
        st.write("## Please upload an image file")
        
if __name__ == "__main__":
    main()
import numpy as np
import io
import streamlit as st
from PIL import Image
import tensorflow as tf
import pickle

st.title("Handwritten Digit Recognition")
st.markdown("Upload an image")



# pickle_in = open('model.pkl', 'rb')
# model = pickle.load(pickle_in)
  

model = tf.keras.models.load_model("model")

test_img = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])
submit = st.button("Predict")

if submit:
    if test_img is not None:
        imagefile = io.BytesIO(test_img.read())
        test_image = Image.open(imagefile)
        test_image = test_image.resize((64, 64))
        test_image = test_image.convert("L")  
        test_image = np.array(test_image) / 255.0  
        test_image = np.expand_dims(test_image, 0) 
        predictions = model.predict(test_image)
        predicted_digit = np.argmax(predictions)

        st.title("Output is "+ str(predicted_digit))
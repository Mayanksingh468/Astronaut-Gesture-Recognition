import os
import streamlit as st

st.write("Step 1")

import tensorflow as tf
st.write("Step 2")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "efficientnetb0_finetuned.keras"
)

st.write(f"Model path: {MODEL_PATH}")
st.write(f"Exists: {os.path.exists(MODEL_PATH)}")

model = tf.keras.models.load_model(MODEL_PATH)

st.write("Step 3 - Model Loaded")
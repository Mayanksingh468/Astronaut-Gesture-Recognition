import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "efficientnetb0_finetuned.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "del","nothing","space"
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Astronaut Hand Gesture Recognition",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 Astronaut Hand Gesture Recognition")

st.success("Model Loaded Successfully")

st.write(
    "Upload an astronaut hand gesture image and get the predicted ASL character."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Resize
        img = image.resize((224, 224))

        # Convert to array
        img_array = np.array(img)

        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess for EfficientNet
        img_array = preprocess_input(img_array)

        # Prediction
        predictions = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(predictions)

        predicted_class = class_names[predicted_index]

        confidence = float(predictions[0][predicted_index]) * 100

        st.success(f"Prediction: {predicted_class}")

        st.info(f"Confidence: {confidence:.2f}%")

        # -----------------------------
        # Top 3 Predictions
        # -----------------------------
        st.subheader("Top 3 Predictions")

        top3_idx = np.argsort(predictions[0])[-3:][::-1]

        for idx in top3_idx:
            st.write(
                f"{class_names[idx]} : {predictions[0][idx] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
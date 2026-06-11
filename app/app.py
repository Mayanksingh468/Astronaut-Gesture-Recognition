import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
model = tf.keras.models.load_model(
    "efficientnetb0_finetuned.keras"
)

# Class labels
class_names = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "del","nothing","space"
]

# App Title
st.title("🚀 Astronaut Hand Gesture Recognition")

st.write(
    "Upload a hand gesture image and get prediction."
)

# Upload Image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open Image
    image = Image.open(uploaded_file)

    # Display Image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing
    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image)

    image = preprocess_input(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    # Prediction
    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # Show Results
    st.success(
        f"Prediction: {class_names[predicted_class]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # Top 3 Predictions
    st.subheader("Top 3 Predictions")

    top3_idx = np.argsort(prediction[0])[-3:][::-1]

    for idx in top3_idx:
        st.write(
            f"{class_names[idx]} : {prediction[0][idx] * 100:.2f}%"
        )
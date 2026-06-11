import os
import tensorflow as tf

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "efficientnetb0_finetuned.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)
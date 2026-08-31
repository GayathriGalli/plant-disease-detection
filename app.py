import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload an image of a plant leaf and the deep learning model "
    "will analyze it and predict the plant disease."
)

MODEL_PATH = Path("model/plant_disease_model.keras")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


model = load_model()

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )

    if model is None:
        st.warning(
            "The trained model is not included in this repository yet. "
            "Add the trained model to model/plant_disease_model.keras "
            "to enable predictions."
        )

    elif st.button("Predict Disease"):
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        st.success(f"Predicted class index: {predicted_index}")
        st.info(f"Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption(
    "Plant Disease Detection using Deep Learning and Computer Vision"
)

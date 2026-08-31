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
CLASS_NAMES_PATH = Path("model/class_names.txt")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        return []

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


model = load_model()
class_names = load_class_names()

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
            "The trained model is not available yet. "
            "Run train_model.py to generate the model."
        )

    elif not class_names:
        st.warning(
            "Disease class labels are not available. "
            "Run train_model.py to generate class_names.txt."
        )

    elif st.button("🔍 Predict Disease"):
        processed_image = preprocess_image(image)

        with st.spinner("Analyzing leaf image..."):
            predictions = model.predict(
                processed_image,
                verbose=0
            )[0]

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index]) * 100

        if predicted_index < len(class_names):
            disease_name = class_names[predicted_index]
        else:
            disease_name = f"Class {predicted_index}"

        disease_name = disease_name.replace("___", " - ")
        disease_name = disease_name.replace("_", " ")

        st.success(f"🌱 Prediction: {disease_name}")
        st.info(f"🎯 Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption(
    "Plant Disease Detection using Deep Learning & Computer Vision"
)

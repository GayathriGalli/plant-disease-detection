import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

MODEL_PATH = Path("model/plant_disease_model.keras")
CLASS_NAMES_PATH = Path("model/class_names.txt")
IMAGE_SIZE = (224, 224)


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "class_names.txt was not found in the model directory."
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_disease(image_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    image = preprocess_image(image_path)

    predictions = model.predict(image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    return {
        "disease": class_names[predicted_index],
        "confidence": confidence
    }


if __name__ == "__main__":
    test_image = "sample_images/test_leaf.jpg"

    result = predict_disease(test_image)

    print("Predicted Disease:", result["disease"])
    print(f"Confidence: {result['confidence'] * 100:.2f}%")

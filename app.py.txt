
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Melanoma CNN Classifier",
    page_icon="🔬",
    layout="centered"
)

st.title("Melanoma Image Classifier")
st.write("Upload an image to get an experimental CNN model prediction.")

# Replace with your actual model filename
MODEL_PATH = "melanoma_model.keras"

# Temporary labels until you confirm Kaggle's class mapping
CLASS_NAMES = None

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model()
    st.success("CNN model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Read the expected image dimensions from the model
try:
    input_shape = model.input_shape
    height = input_shape[1] or 224
    width = input_shape[2] or 224
except (AttributeError, TypeError, IndexError):
    height, width = 224, 224

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=350)

    if st.button("Run prediction", type="primary"):
        resized = image.resize((width, height))
        array = np.asarray(resized, dtype=np.float32)

        # Keep this only if your training used /255.0
        array = array / 255.0
        array = np.expand_dims(array, axis=0)

        try:
            output = np.asarray(model.predict(array, verbose=0)).ravel()

            if output.size == 1:
                score = float(output[0])
                st.write(f"Model output score: {score:.4f}")

                st.info(
                    "Binary model output. A threshold of 0.5 is a common "
                    "example, but verify your model's output activation "
                    "and which class is encoded as 1 before interpreting it."
                )

                if 0 <= score <= 1:
                    predicted_index = int(score >= 0.5)
                    st.write(f"Predicted class index: {predicted_index}")

            else:
                predicted_index = int(np.argmax(output))
                st.write(
                    f"Predicted class index: {predicted_index}"
                )
                st.write(
                    f"Output score: {float(output[predicted_index]):.4f}"
                )

                if CLASS_NAMES and predicted_index < len(CLASS_NAMES):
                    st.subheader(
                        f"Predicted class: {CLASS_NAMES[predicted_index]}"
                    )
                else:
                    st.caption(
                        "Class names are placeholders until the Kaggle "
                        "class mapping is confirmed."
                    )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.warning(
    "This is an educational machine-learning demo, not a medical "
    "diagnostic tool. Seek assessment from a qualified healthcare "
    "professional for any concerning skin lesion."
)


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from data_preprocessing import preprocess_image


# Function to preprocess a single image for prediction
def preprocess_input_image(image_path, bbox=None, target_size=(32, 32)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if bbox:
        x, y, w, h = bbox.strip('[]').split(',')
        x, y, w, h = float(x), float(y), float(w), float(h)
        bbox_image = image[int(y):int(y + h), int(x):int(x + w)]
    else:
        bbox_image = image

    preprocessed_image = preprocess_image(bbox_image, target_size)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    return preprocessed_image


# Function to create and fit the label encoder
def create_label_encoder(annot_path, num_rows=None):
    annotations = pd.read_csv(annot_path, nrows=num_rows)
    labels = annotations['utf8_string'].dropna().astype(str).values
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder


# Function to make prediction and decode text
def predict_text(image_path, model, label_encoder, bbox=None):
    preprocessed_image = preprocess_input_image(image_path, bbox)
    prediction = model.predict(preprocessed_image)
    predicted_label = np.argmax(prediction, axis=-1)
    predicted_text = label_encoder.inverse_transform(predicted_label)[0]

    return predicted_text


if __name__ == "__main__":
    model_path = "ocr_model.h5"
    annot_path = "annot_filtered.csv"  # Path to your annotations CSV file
    image_path = "SERKAN.png"  # Path to the image you want to predict
    num_rows = 25000  # Number of rows to read for creating the label encoder

    # Load the trained model
    model = load_model(model_path)

    # Create and fit the label encoder
    label_encoder = create_label_encoder(annot_path, num_rows)

    # Predict text from the image
    predicted_text = predict_text(image_path, model, label_encoder)

    print(f"Predicted Text: {predicted_text}")

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import editdistance

# Load data loading and preprocessing functions
from data_loading import load_data_new
from data_preprocessing import preprocess_data_new
from model_training import CER_WER_Callback, create_model

# Define a function to evaluate the model
def evaluate_model(model_path, annot_path, images_dir, num_rows=1000):
    # Load and preprocess data
    dataset, vocab, max_len = load_data_new(annot_path, images_dir, num_rows)
    images, labels = preprocess_data_new(dataset)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    encoded_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes)

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Make predictions on the entire dataset
    predictions = model.predict(images)
    pred_labels = np.argmax(predictions, axis=-1)
    true_labels = np.argmax(encoded_labels, axis=-1)

    pred_texts = label_encoder.inverse_transform(pred_labels)
    true_texts = label_encoder.inverse_transform(true_labels)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f'Accuracy: {accuracy:.4f}')

    # Calculate CER
    cer = np.mean([editdistance.eval(pred, true) / len(true) for pred, true in zip(pred_texts, true_texts)])
    print(f'CER: {cer:.4f}')

    # Calculate WER
    wer = np.mean([editdistance.eval(pred.split(), true.split()) / len(true.split()) for pred, true in zip(pred_texts, true_texts)])
    print(f'WER: {wer:.4f}')

    return accuracy, cer, wer

if __name__ == "__main__":
    annot_path = "annot_filtered.csv"
    images_dir = "train/"
    model_path = "ocr_model.h5"
    num_rows = 25000  # Specify the number of rows to read for evaluation

    accuracy, cer, wer = evaluate_model(model_path, annot_path, images_dir, num_rows)
    print(f"Evaluation complete. Accuracy: {accuracy:.4f}, CER: {cer:.4f}, WER: {wer:.4f}")

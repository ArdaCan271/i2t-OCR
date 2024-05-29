import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import editdistance  # To calculate CER and WER

# Load data loading and preprocessing functions
from data_loading import load_data
from data_loading import load_data_new

from data_preprocessing import preprocess_data
from data_preprocessing import preprocess_data_new

# Define the custom callback to compute CER and WER
class CER_WER_Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, label_encoder):
        super().__init__()
        self.validation_data = validation_data
        self.label_encoder = label_encoder

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        predictions = self.model.predict(X_val)
        pred_labels = np.argmax(predictions, axis=-1)
        true_labels = np.argmax(y_val, axis=-1)

        pred_texts = self.label_encoder.inverse_transform(pred_labels)
        true_texts = self.label_encoder.inverse_transform(true_labels)

        cer = np.mean([editdistance.eval(pred, true) / len(true) for pred, true in zip(pred_texts, true_texts)])
        wer = np.mean([editdistance.eval(pred.split(), true.split()) / len(true.split()) for pred, true in zip(pred_texts, true_texts)])

        print(f' - CER: {cer:.4f} - WER: {wer:.4f}')


def create_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    annot_path = "annot_filtered.csv"
    images_dir = "train/"
    num_rows = 25000  # Specify the number of rows to read

    # Load and preprocess data
    train_dataset, train_vocab, max_train_len = load_data_new(annot_path, images_dir, num_rows)
    images, labels = preprocess_data_new(train_dataset)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    encoded_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

    # Define input shape
    input_shape = X_train.shape[1:]  # e.g., (32, 32, 3)

    # Create and compile model
    model = create_model(input_shape, num_classes)

    # Define custom callback for CER and WER
    cer_wer_callback = CER_WER_Callback(validation_data=(X_val, y_val), label_encoder=label_encoder)

    # Train the model with the custom callback
    epochs = 100
    batch_size = 64

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[cer_wer_callback])

    # Save the model
    model.save("ocr_model.h5")

    print("Model training complete and saved as 'ocr_model.h5'.")

import numpy as np
import cv2
import os
from tqdm import tqdm
from joblib import Parallel, delayed


def preprocess_image(image, target_size=(32, 32)):
    image = cv2.resize(image, target_size)                          # Resize the image to the target size
    image = image / 255.0                                           # Normalize the image
    return image


def process_single_image(image_path, utf8_string, bbox):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    x, y, w, h = bbox.strip('[]').split(',')
    x, y, w, h = float(x), float(y), float(w), float(h)
    
    height, width = image.shape[:2]                                 # Ensure the bounding box is within image bounds
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return None, None                                           # Skip this image
        
    bbox_image = image[int(y):int(y + h), int(x):int(x + w)]
    preprocessed_image = preprocess_image(bbox_image)
    return preprocessed_image, utf8_string


def preprocess_data_new(dataset):
    results = Parallel(n_jobs=-1)(
        delayed(process_single_image)(image_path, utf8_string, bbox) for image_path, utf8_string, bbox in
        tqdm(dataset, desc="Processing Data"))

    # Filter out None results
    results = [result for result in results if result[0] is not None]

    images, labels = zip(*results)
    images = np.array(images)
    labels = np.array(labels)

    # Debugging: print shapes
    print(f"Image batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")

    return images, labels


def preprocess_data(data):
    images = []
    labels = []

    for idx, (bbox_image, utf8_string) in enumerate(data):
        print("idx: ", idx)
        print("bbox_image: ", bbox_image)
        print("utf8_string: ", utf8_string)
        preprocessed_image = preprocess_image(bbox_image)
        images.append(preprocessed_image)
        labels.append(utf8_string)

    images = np.array(images)
    labels = np.array(labels)

    # Debugging: print shapes
    print(f"Image batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")

    return images, labels


if __name__ == "__main__":
    from data_loading import load_data
    from data_loading import load_data_new

    annot_path = "annot.csv"
    images_dir = "train/"
    num_rows = 100  # Specify the number of rows to read
    train_dataset, train_vocab, max_train_len = load_data_new(annot_path, images_dir, num_rows=num_rows)
    # data = load_data(annot_path, images_dir, num_rows=num_rows)
    # images, labels = preprocess_data(data)
    images, labels = preprocess_data_new(train_dataset)
    # print(f"Preprocessed {len(images)} images and saved them.")

import pandas as pd
import cv2
import os
from tqdm import tqdm


def load_data(annot_path, images_dir, save_dir=None, num_rows=None):
    annotations = pd.read_csv(annot_path, nrows=num_rows)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    data = []

    for idx, row in tqdm(annotations.iterrows(), total=annotations.shape[0], desc="Loading data"):
        image_id = row['image_id']
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        image = cv2.imread(image_path)
        x, y, w, h = row['bbox'].strip('[]').split(',')
        x, y, w, h = float(x), float(y), float(w), float(h)
        bbox_image = image[int(y):int(y + h), int(x):int(x + w)]
        utf8_string = row['utf8_string']
        data.append((bbox_image, utf8_string))

    return data


def load_data_new(annot_path, images_dir, num_rows=100):
    dataset, vocab, max_len = [], set(), 0
    annotations_df = pd.read_csv(annot_path).head(num_rows)
    
    for index, row in tqdm(annotations_df.iterrows(), total=annotations_df.shape[0], desc="Loading data"):
        image_id = row['image_id']
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        label = str(row['utf8_string'])
        bbox = row['bbox']
        if label == 'nan':
            continue
        dataset.append([image_path, label, bbox])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

    return dataset, vocab, max_len


if __name__ == "__main__":
    annot_path = "annot.csv"
    images_dir = "train/"
    num_rows = 100  # Specify the number of rows to read
    train_dataset, train_vocab, max_train_len = load_data_new(annot_path, images_dir, num_rows=num_rows)
    # data = load_data(annot_path, images_dir, num_rows=num_rows)
    print(f"Loaded {len(train_dataset)} bounding boxes")

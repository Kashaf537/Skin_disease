r"""
Skin Disease CNN — Testing / Inference Script
================================================
Run this AFTER train_fixed.py has produced:
    models/final_model.keras
    models/class_names.json

Two modes:

  1) Full evaluation on the held-out test_set (classification report +
     confusion matrix + a list of misclassified files to eyeball):
        python test.py

  2) Predict on one or more new images you give it directly (confidence
     bar chart saved per image, top-3 printed to console):
        python test.py --image path\to\img1.jpg path\to\img2.jpg
"""

import os
import argparse
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix


# ==================== CONFIGURATION ====================
# keep these in sync with train_fixed.py
class Config:
    TEST_PATH = r"D:\Skin disease\test_set"
    MODEL_DIR = "models"
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32


# ==================== LOADING ====================
def load_class_names():
    path = os.path.join(Config.MODEL_DIR, "class_names.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run train_fixed.py first.")
    with open(path) as f:
        return json.load(f)


def load_trained_model():
    path = os.path.join(Config.MODEL_DIR, "final_model.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run train_fixed.py first.")
    return keras.models.load_model(path)


# ==================== PREPROCESSING (must match training) ====================
def clahe_preprocess(img_rgb_uint8):
    """Same CLAHE contrast normalization used in train_fixed.py. Must
    stay identical to training preprocessing or predictions will be off."""
    lab = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)


def load_and_preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, Config.IMG_SIZE)
    img = clahe_preprocess(img)
    return img.astype(np.float32)  # model's Rescaling layer handles [-1,1] scaling internally


def _tf_clahe(image, label):
    image = tf.py_function(
        func=lambda im: clahe_preprocess(im.numpy().astype(np.uint8)).astype(np.float32),
        inp=[image], Tout=tf.float32,
    )
    image.set_shape([Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3])
    return image, label


# ==================== MODE 1: full test_set evaluation ====================
def run_full_test_evaluation(model, class_names):
    print(f"\nEvaluating on: {Config.TEST_PATH}")

    test_ds = keras.utils.image_dataset_from_directory(
        Config.TEST_PATH, image_size=Config.IMG_SIZE, batch_size=Config.BATCH_SIZE,
        label_mode="categorical", shuffle=False,
    )
    file_paths = test_ds.file_paths  # grab before .map()/.batch() reshuffles structure
    test_ds_c = (test_ds.unbatch()
                 .map(_tf_clahe, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(Config.BATCH_SIZE))

    loss, acc = model.evaluate(test_ds_c, verbose=0)
    print(f"\nTest accuracy: {acc:.4f} | Test loss: {loss:.4f}")

    y_true, y_pred, all_probs = [], [], []
    for images, labels in test_ds_c:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        all_probs.extend(preds)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (test_set)')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    out_path = os.path.join(Config.MODEL_DIR, 'test_confusion_matrix.png')
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"\nConfusion matrix saved to {out_path}")

    print("\nMisclassified samples (up to 15, for manual review):")
    shown = 0
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p and shown < 15:
            fname = os.path.basename(file_paths[i])
            print(f"  {fname:40s} true={class_names[t]:28s} pred={class_names[p]:28s} conf={all_probs[i][p]:.2f}")
            shown += 1
    if shown == 0:
        print("  (none — perfect on this run)")


# ==================== MODE 2: predict on new images ====================
def run_single_image_predictions(model, class_names, image_paths):
    for path in image_paths:
        print(f"\n{'=' * 60}\n{path}")
        img = load_and_preprocess_image(path)
        batch = np.expand_dims(img, axis=0)
        probs = model.predict(batch, verbose=0)[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        print(f"Predicted: {class_names[top3_idx[0]]}  ({probs[top3_idx[0]] * 100:.1f}% confidence)")
        print("Top 3:")
        for idx in top3_idx:
            print(f"  {class_names[idx]:35s} {probs[idx] * 100:5.1f}%")

        plt.figure(figsize=(8, 4))
        bars = plt.barh(class_names, probs * 100, color="#4C72B0")
        bars[top3_idx[0]].set_color("#55A868")
        plt.xlabel("Confidence (%)")
        plt.title(os.path.basename(path))
        plt.gca().invert_yaxis()
        plt.tight_layout()
        out_name = os.path.splitext(os.path.basename(path))[0] + "_prediction.png"
        out_path = os.path.join(Config.MODEL_DIR, out_name)
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"Saved confidence chart -> {out_path}")


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="Test the trained skin disease classifier")
    parser.add_argument("--image", nargs="+", help="One or more image paths to predict on")
    args = parser.parse_args()

    class_names = load_class_names()
    model = load_trained_model()
    print(f"Loaded model. Classes: {class_names}")

    if args.image:
        run_single_image_predictions(model, class_names, args.image)
    else:
        run_full_test_evaluation(model, class_names)


if __name__ == "__main__":
    main()
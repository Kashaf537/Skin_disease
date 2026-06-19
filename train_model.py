"""
Skin Disease CNN — FIXED pipeline
==================================
Same overall structure as your original script, with the issues fixed:

  1. Flatten() -> GlobalAveragePooling2D()   (removes ~8M useless params)
  2. From-scratch CNN -> MobileNetV2 transfer learning (auto-fallback if offline)
  3. Validation generator pointed at TEST_PATH -> real train/val split,
     TEST_PATH is now only touched once, at the very end
  4. Deprecated/flaky ImageDataGenerator -> stable tf.data pipeline
  5. Added real OpenCV preprocessing (CLAHE contrast normalization) via
     a proper preprocessing function, wired into both training AND
     single-image inference (so train/inference stay consistent)
  6. Added class_weight for the mild class imbalance (80-136 imgs/class)
  7. class_names are now saved to disk and loaded for inference, instead
     of a hardcoded list that can silently drift out of sync

Usage:
    python train_fixed.py
(edit Config.TRAIN_PATH / Config.TEST_PATH below, same as before)
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
class Config:
    TRAIN_PATH = r"D:\Skin disease\train_set"
    TEST_PATH = r"D:\Skin disease\test_set"
    MODEL_DIR = "models"

    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 30                 # head training (frozen backbone)
    FINE_TUNE_EPOCHS = 15       # extra epochs with backbone unfrozen
    VAL_SPLIT = 0.15            # carved out of TRAIN_PATH, TEST_PATH stays untouched

    NUM_CLASSES = 8
    LEARNING_RATE = 0.001
    FINE_TUNE_LR = 1e-5
    SEED = 42


# ==================== DATA EXPLORATION ====================
def explore_dataset():
    print("=" * 50)
    print("DATASET EXPLORATION")
    print("=" * 50)
    for label, path in [("Training", Config.TRAIN_PATH), ("Test", Config.TEST_PATH)]:
        print(f"\n{label} classes and image counts:")
        for class_name in sorted(os.listdir(path)):
            class_dir = os.path.join(path, class_name)
            if os.path.isdir(class_dir):
                print(f"  - {class_name}: {len(os.listdir(class_dir))} images")
    print("\n" + "=" * 50)


# ==================== OPENCV PREPROCESSING ====================
def clahe_preprocess(image_uint8):
    """CLAHE on the L channel of LAB space: normalizes contrast/lighting
    across the clinical photos (which vary a lot in lighting/camera),
    instead of just relying on naive ./255 rescaling."""
    img = image_uint8.numpy().astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return rgb.astype(np.float32)


def _tf_clahe(image, label):
    image = tf.py_function(func=clahe_preprocess, inp=[image], Tout=tf.float32)
    image.set_shape([Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3])
    return image, label


def _apply_clahe(ds, batch_size, shuffle_buffer=None):
    ds = ds.unbatch().map(_tf_clahe, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer, seed=Config.SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ==================== DATA LOADING ====================
def load_data():
    print("\nLoading and preparing data (stable tf.data pipeline)...")

    train_raw = keras.utils.image_dataset_from_directory(
        Config.TRAIN_PATH, image_size=Config.IMG_SIZE, batch_size=Config.BATCH_SIZE,
        validation_split=Config.VAL_SPLIT, subset="training", seed=Config.SEED,
        label_mode="categorical",
    )
    class_names = train_raw.class_names

    val_raw = keras.utils.image_dataset_from_directory(
        Config.TRAIN_PATH, image_size=Config.IMG_SIZE, batch_size=Config.BATCH_SIZE,
        validation_split=Config.VAL_SPLIT, subset="validation", seed=Config.SEED,
        label_mode="categorical",
    )
    test_raw = keras.utils.image_dataset_from_directory(
        Config.TEST_PATH, image_size=Config.IMG_SIZE, batch_size=Config.BATCH_SIZE,
        label_mode="categorical", shuffle=False,
    )

    # grab integer labels (for class_weight) before CLAHE remaps the dataset
    train_labels = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in train_raw])

    train_ds = _apply_clahe(train_raw, Config.BATCH_SIZE, shuffle_buffer=200)
    val_ds = _apply_clahe(val_raw, Config.BATCH_SIZE)
    test_ds = _apply_clahe(test_raw, Config.BATCH_SIZE)

    print(f"\nClass mapping: {dict(enumerate(class_names))}")
    print(f"Number of classes: {len(class_names)}")

    return train_ds, val_ds, test_ds, class_names, train_labels


# ==================== MODEL CREATION ====================
def create_cnn_model(num_classes):
    """MobileNetV2 transfer learning + GlobalAveragePooling head.
    Tries ImageNet weights first; if offline, falls back to random init
    with a clear warning (still far fewer params than Flatten+Dense)."""
    print("\nCreating CNN model (MobileNetV2 transfer learning)...")

    inputs = keras.Input(shape=(*Config.IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomContrast(0.1)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)  # MobileNetV2 expects [-1, 1]

    pretrained = True
    try:
        base = keras.applications.MobileNetV2(
            input_shape=(*Config.IMG_SIZE, 3), include_top=False, weights="imagenet"
        )
    except Exception as e:
        print(f"\n[WARN] Could not download ImageNet weights ({e}).")
        print("[WARN] Training MobileNetV2 FROM SCRATCH instead.")
        print("[WARN] Run this on a machine with internet access for real transfer learning.\n")
        pretrained = False
        base = keras.applications.MobileNetV2(
            input_shape=(*Config.IMG_SIZE, 3), include_top=False, weights=None
        )

    base.trainable = False  # frozen for phase 1 regardless; phase 2 unfreezes if pretrained
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)   # <- was Flatten(); this is the key fix
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model, base, pretrained


# ==================== MODEL TRAINING ====================
def train_model(model, base, pretrained, train_ds, val_ds, class_weight):
    print("\nStarting training...")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        # NOTE: save_weights_only=True sidesteps a known Keras bug where
        # ModelCheckpoint(save_best_only=True) + the native .keras format
        # raises "argument(s) not supported with the native Keras format:
        # ['options']" on some Keras patch versions (tensorflow/tensorflow#61999).
        # EarlyStopping(restore_best_weights=True) below + the explicit
        # model.save() at the end of this function already give you the
        # best full model — this checkpoint is just a mid-training safety net.
        ModelCheckpoint(filepath=os.path.join(Config.MODEL_DIR, 'best_model.weights.h5'),
                         monitor='val_accuracy', save_best_only=True,
                         save_weights_only=True, verbose=1),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=Config.EPOCHS,
        class_weight=class_weight, callbacks=callbacks, verbose=1,
    )

    # Phase 2: fine-tune — only meaningful if real ImageNet weights loaded
    if pretrained and Config.FINE_TUNE_EPOCHS > 0:
        print("\nUnfreezing backbone for fine-tuning...")
        base.trainable = True
        model.compile(optimizer=Adam(learning_rate=Config.FINE_TUNE_LR),
                       loss="categorical_crossentropy", metrics=["accuracy"])
        history_ft = model.fit(
            train_ds, validation_data=val_ds, epochs=Config.FINE_TUNE_EPOCHS,
            class_weight=class_weight, callbacks=callbacks, verbose=1,
        )
        for k in history.history:
            history.history[k] += history_ft.history[k]

    model.save(os.path.join(Config.MODEL_DIR, 'final_model.keras'))
    print("\nModel saved successfully!")
    return history


# ==================== VISUALIZATION ====================
def plot_training_history(history):
    print("\nGenerating training plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_DIR, 'training_history.png'), dpi=100)
    plt.close()

    print(f"\nFinal Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")


# ==================== EVALUATION (on the held-out TEST set, touched once) ====================
def evaluate_model(model, test_ds, class_names):
    print("\nEvaluating model on held-out TEST set...")

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (TEST set)')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_DIR, 'confusion_matrix.png'), dpi=100)
    plt.close()


# ==================== PREDICTION ====================
def predict_single_image(image_path, model_path=None):
    """Predict disease from a single image. Applies the SAME CLAHE
    preprocessing used in training, so train/inference stay consistent."""
    if model_path is None:
        # final_model.keras is the full model (architecture + weights),
        # saved after EarlyStopping already restored the best epoch's
        # weights. best_model.weights.h5 (the mid-training checkpoint)
        # is weights-only, so it needs the architecture rebuilt first —
        # final_model.keras is simpler for plain inference.
        model_path = os.path.join(Config.MODEL_DIR, 'final_model.keras')
    if not os.path.exists(model_path):
        print("Model not found! Please train the model first.")
        return None, None

    class_names_path = os.path.join(Config.MODEL_DIR, 'class_names.json')
    with open(class_names_path) as f:
        class_names = json.load(f)

    model = keras.models.load_model(model_path)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, Config.IMG_SIZE)

    # apply CLAHE the same way as training preprocessing did
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    img_clahe = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)

    img_batch = np.expand_dims(img_clahe.astype(np.float32), axis=0)  # model rescales internally

    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    print(f"\nPredicted Disease: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("\nAll Probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {predictions[0][i]:.4f}")

    return class_names[predicted_class], confidence


# ==================== MAIN EXECUTION ====================
def main():
    print("=" * 50)
    print("SKIN DISEASE DETECTION CNN — FIXED PIPELINE")
    print("=" * 50)

    explore_dataset()
    train_ds, val_ds, test_ds, class_names, train_labels = load_data()

    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    with open(os.path.join(Config.MODEL_DIR, 'class_names.json'), 'w') as f:
        json.dump(class_names, f, indent=2)

    weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight = dict(enumerate(weights))
    print(f"\nClass weights (for imbalance): {class_weight}")

    model, base, pretrained = create_cnn_model(len(class_names))
    history = train_model(model, base, pretrained, train_ds, val_ds, class_weight)
    plot_training_history(history)
    evaluate_model(model, test_ds, class_names)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"pretrained_imagenet_weights_used = {pretrained}")
    print(f"Model + plots saved in: {Config.MODEL_DIR}/")


if __name__ == "__main__":
    main()

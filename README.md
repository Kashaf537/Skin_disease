# 🩺 DermAI Clinical Assistant

A hybrid edge + cloud skin-condition triage assistant. A local CNN gives an
instant classification across 8 skin conditions; a cloud vision-language
model independently double-checks that result and adds plain-language
causes/treatment context when it agrees.

> ⚠️ **Medical Disclaimer:** This project is for educational/portfolio
> purposes only. It is **not** a medical device and must not be used for
> real diagnosis or treatment decisions. Always consult a licensed
> dermatologist or physician for any skin concern.

---

## Features

- 🧠 **8-class skin condition classifier** — bacterial (cellulitis,
  impetigo), fungal (athlete's foot, nail fungus, ringworm), parasitic
  (cutaneous larva migrans), and viral (chickenpox, shingles) conditions
- 🎛️ **OpenCV CLAHE preprocessing** — normalizes lighting/contrast across
  inconsistent clinical photos before they reach the model
- ☁️ **Cloud "second opinion"** — a vision-language model (via Groq)
  independently reviews the same image, confirms or overrides the local
  category, and explains *why*
- 📋 **Auto-generated clinical context** — causes/etiology and general
  treatment information for verified matches (informational only)
- 🛟 **Graceful degradation** — if the cloud call fails or is slow, the app
  falls back to the local model gated by a configurable confidence
  threshold, so there's always a usable result
- 📊 **Live metrics** — inference confidence, cloud round-trip latency,
  and a top-3 class probability breakdown on every scan
- 🎨 **Custom dark clinical dashboard UI**
- 🧪 **Standalone test/evaluation script** — full test-set classification
  report + confusion matrix, or predict on arbitrary new images
- 🏋️ **Transfer-learning training pipeline** — MobileNetV2 backbone,
  two-phase (frozen → fine-tuned) training, class-weighted for the dataset's
  mild imbalance

## Tech Stack

| Layer | Tools |
|---|---|
| Model | TensorFlow / Keras — MobileNetV2 (transfer learning) + GlobalAveragePooling2D head |
| Image preprocessing | OpenCV (CLAHE contrast normalization) |
| Cloud verification | Groq API — Llama 4 Scout vision-language model |
| Structured LLM output | Pydantic |
| Web UI | Streamlit |
| Config | python-dotenv |
| Evaluation | scikit-learn, matplotlib, seaborn |

## Project Structure

```
.
├── train_fixed.py        # Training: MobileNetV2 transfer learning + CLAHE + class weights
├── test.py                # Standalone test-set evaluation OR single-image prediction
├── skin_disease_app.py    # Streamlit app: local model + Groq cloud verification
├── requirements.txt
├── .env.example
└── models/                 # created by train_fixed.py
    ├── final_model.keras          # full trained model (used by test.py + the app)
    ├── best_model.weights.h5      # mid-training checkpoint (weights only)
    ├── class_names.json
    ├── training_history.png
    └── confusion_matrix.png
```

## How It Works

1. User uploads a photo through the Streamlit UI.
2. The image is resized and passed through CLAHE preprocessing (OpenCV) to
   normalize contrast/lighting.
3. The local MobileNetV2 model predicts a class + confidence score across
   the 8 known categories — this is instant and runs on-device.
4. In parallel, the original image is sent to Groq's Llama 4 Scout vision
   model, which independently identifies the condition, decides whether it
   actually falls within the 8 supported categories, and (if so) returns a
   description, likely causes, and general treatment information as
   structured JSON.
5. If the cloud call errors out, the app doesn't break — it falls back to
   the local-only prediction, gated by a confidence threshold so low-
   confidence guesses are flagged rather than presented as fact.
6. Results, confidence metrics, and a top-3 breakdown are rendered on a
   dashboard-style UI.

## Setup

```bash
# 1. Clone/download this project, then create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Groq API key
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# then edit .env and paste your key from https://console.groq.com
```

Update the `Config` class at the top of `train_fixed.py`, `test.py`, and
`skin_disease_app.py` to match your local paths (dataset location, model
output directory, image size) if you change them from the defaults.

## Usage

### 1. Train the model
```bash
python train_fixed.py
```
Trains MobileNetV2 (ImageNet weights, frozen → fine-tuned) on `train_set/`,
validates on a split carved out of it, and reports final metrics on the
untouched `test_set/`. Saves the model + plots to `models/`.

### 2. Test / evaluate
```bash
# Full evaluation on the held-out test set (classification report + confusion matrix)
python test.py

# Predict on specific new images instead
python test.py --image "path\to\image1.jpg" "path\to\image2.jpg"
```

### 3. Run the web app
```bash
streamlit run skin_disease_app.py
```
Opens the dashboard in your browser. Requires `models/final_model.keras`
and `models/class_names.json` to exist (run training first) and a valid
`GROQ_API_KEY` in `.env` for the cloud verification step.

## Model Details

- **Architecture:** MobileNetV2 backbone (ImageNet-pretrained) →
  GlobalAveragePooling2D → Dropout → Dense(128, L2-regularized) →
  Dropout → Dense(8, softmax)
- **Input size:** 128×128, normalized to [-1, 1] internally
- **Preprocessing:** CLAHE on the L channel of LAB color space, applied
  identically at train and inference time
- **Training:** two-phase — head-only with frozen backbone, then full
  fine-tuning at a lower learning rate — with early stopping, LR reduction
  on plateau, and class weighting for the ~80–136-images-per-class
  imbalance
- **Classes (8):** BA-cellulitis, BA-impetigo, FU-athlete-foot,
  FU-nail-fungus, FU-ringworm, PA-cutaneous-larva-migrans, VI-chickenpox,
  VI-shingles

See `models/training_history.png` and `models/confusion_matrix.png` after
training for accuracy/loss curves and per-class performance.

dataset:https://drive.google.com/drive/folders/1mIPkKlG0pmDJU0ConGgqcJ3UUHQOGQz8?usp=drive_link




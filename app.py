from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from gradcam import make_gradcam_heatmap, overlay_heatmap
import cv2

from utils.preprocess import preprocess_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "model/best_model.keras"

def build_placeholder_model(input_shape=(224, 224, 3), num_classes=4):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same", name="activation_17")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print(f"WARNING: No model found at {MODEL_PATH}. Using a placeholder model instead.")
    model = build_placeholder_model()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        model.save(MODEL_PATH)
        print(f"Placeholder model saved to {MODEL_PATH}")
    except Exception as e:
        print("WARNING: Could not save placeholder model:", e)

print(model.summary())
for layer in model.layers:
    print(layer.name)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img, original_img = preprocess_image(file)

    # 🔥 DEBUG (keep for now)
    print("IMG SHAPE:", img.shape)
    print("IMG MIN/MAX:", img.min(), img.max())

    pred = model.predict(img)

    idx = int(np.argmax(pred))
    confidence = float(np.max(pred))

    # 🔥 Grad-CAM
    heatmap = make_gradcam_heatmap(img, model, "activation_17")

    print("HEATMAP MIN/MAX:", heatmap.min(), heatmap.max())

    # 🔥 Overlay
    overlay = overlay_heatmap(
        original_img,
        heatmap
    )

    # 🔥 SAFE conversion (important fix)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Ensure folder exists
    os.makedirs("static/gradcam", exist_ok=True)

    gradcam_path = os.path.join(
        "static",
        "gradcam",
        "gradcam.jpg"
    )

    cv2.imwrite(gradcam_path, overlay)

    return jsonify({
        "prediction": class_names[idx],
        "confidence": confidence,
        "gradcam": "/static/gradcam/gradcam.jpg"
    })

@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# RUN SERVER (production-safe config)
# -----------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False  # IMPORTANT: disable debug in production
    )
# frame_duplication_cnn.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# -------------------------
# Function: Extract frames
# -------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))  # Resize for model input
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames

# -------------------------
# Function: Generate dataset
# -------------------------
def generate_dataset():
    # Replace with your actual dataset pipeline
    X_train = np.random.rand(100, 64, 64, 3)  # 100 frames, 64x64 RGB
    y_train = np.random.randint(0, 2, 100)    # 0 = normal, 1 = duplicated
    X_test = np.random.rand(30, 64, 64, 3)
    y_test = np.random.randint(0, 2, 30)
    return X_train, y_train, X_test, y_test

# -------------------------
# Function: Build CNN model
# -------------------------
def build_model(input_shape=(64, 64, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    # Load/generate dataset
    X_train, y_train, X_test, y_test = generate_dataset()

    # Build and train the model
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

    # Predict on test set
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

    # Compute metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Plot metrics
    plt.bar(['Precision', 'Recall', 'F1-Score'], [precision, recall, f1], color=['blue', 'green', 'orange'])
    plt.title('Duplicate Frame Detection Metrics')
    plt.ylabel('Score')
    for i, v in enumerate([precision, recall, f1]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.ylim(0, 1)
    plt.show()

# Video-Frame-Duplication-Detection-CNN
Frame duplication, a common video tampering technique, involves copying and reinserting frames within footage. Convolutional Neural Networks (CNNs) are deep learning models designed for visual data, using convolutional layers, ReLU activations, pooling layers, and fully connected layers to extract features and generate predictions.

    numpy
    opencv-python
    tensorflow
    scikit-learn
    matplotlib

# Video Frame Duplication Detection using CNN (VFD Dataset)

This project uses a Convolutional Neural Network (CNN) to detect duplicated frames in videos. The dataset used is the **VFD dataset**, part of the **SULFA project**.

## Dataset
The VFD dataset should be structured as:

data/VFD/
normal/ # videos with normal frames
duplicated/ # videos with duplicated frames


## Features
- Extract frames from videos
- Train CNN to detect duplicated frames
- Evaluate model using Precision, Recall, F1-Score
- Plot metrics for visualization

## Usage
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/video-duplication-detection.git
cd video-duplication-detection

2. Install dependencies:
pip install -r requirements.txt

3. Place the VFD dataset in data/VFD/.
4. Run the script:
python frame_duplication_cnn.py

Notes

The model resizes frames to 64x64 for faster training. Adjust size in extract_frames() if needed.
Dataset splitting is 80% train, 20% test.
You can adjust the CNN architecture and hyperparameters for better performance.


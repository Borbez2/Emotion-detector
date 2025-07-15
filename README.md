# Facial-Emotion-Recognition
This project detects facial expressions from live webcam feed using a deep neural network trained on the FER2013 dataset. It predicts 7 emotions and displays real-time labels, confidence bars, and FPS.

## Features:
Trained CNN model with convolution, pooling, dropout, and batch normalization

Real-time facial detection using OpenCV's DNN module

Live webcam emotion recognition with bounding boxes and confidence labels

Visual bar chart showing probability distribution across all emotions

FPS counter

Quit instruction shown on startup (press Q to exit)

## Setup:

pip install tensorflow numpy matplotlib opencv-python

Make sure you have the following files:

emotion_model.keras (your trained model)

models/deploy.prototxt and models/res10_300x300_ssd_iter_140000.caffemodel (face detection model)

To Run:

python facial_recognition.py
Ensure your webcam is connected (If you have multiple webcams, edit the digit on line 16 to cycle though which webcam is used)

The script will detect faces and display the predicted emotion above them

Press q to quit the window


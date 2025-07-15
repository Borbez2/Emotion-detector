# Facial-Emotion-Recognition

This project detects facial expressions from live webcam feed using a deep neural network trained on the FER2013 dataset. It predicts 7 emotions and displays real-time labels, confidence bars, and FPS. Then integrating with an LLM it will provide real time feedback.

## Features

- Trained CNN model with convolution, pooling, dropout, and batch normalization  
- Real-time facial detection using OpenCV's DNN module  
- Live webcam emotion recognition with bounding boxes and confidence labels  
- Visual bar chart showing probability distribution across all emotions  
- FPS counter  
- Quit instruction shown briefly on startup (press Q to exit)  
- Planned LLM integration to respond based on detected emotion  

## Setup

1. Download the dataset from https://www.kaggle.com/datasets/msambare/fer2013  
2. Unzip it and place the folder in the project's root directory  
3. Install dependencies:  
   ```
   pip install tensorflow numpy matplotlib opencv-python
   ```
4. Make sure the following files exist:  
   - emotion_model.keras (your trained model, use train_model.py to create it)  
   - models/deploy.prototxt  
   - models/res10_300x300_ssd_iter_140000.caffemodel

## To Run

```
python facial_recognition.py
```

Make sure your webcam is connected.  
If you have more than one, change the digit on `VideoCapture()` in the script to switch between devices.

Press `q` to exit the window.

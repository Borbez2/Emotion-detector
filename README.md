# Real-Time Facial Emotion Recognition with LLM-Based Conversational Feedback

This project detects facial expressions from live webcam feed using a deep neural network trained on the FER2013 dataset. It predicts 7 emotions and displays real-time labels, confidence bars, and FPS. Then integrating with an LLM it will provide real time feedback.

## Features

- Real-time emotion detection with OpenCV and a CNN trained on FER2013  
- Bounding boxes and emotion labels over each detected face  
- Live bar chart showing confidence for all 7 emotions  
- FPS counter overlay  
- Simple quit instruction on startup (`Q` to exit)  
- Emotion-based terminal chat with a local LLM therapist

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
5.  Install and run [Ollama](https://ollama.com) and pull your preferred local LLM (like Deepseek):
```
ollama pull deepseek
ollama serve
```
## To Run

```
python3 facial_recognition.py
```


Once running:
- Webcam feed will open with detection overlays
- Your terminal will prompt for chat messages based on detected emotions
- Type your message and hit Enter to receive an LLM response
- Press `q` to exit the webcam window

---

### Notes

- All interaction with the LLM happens in the terminal for better performance  
- You can modify the prompt behavior in `llm.py`  
- The app works offline if your model is running locally with Ollama
- If you have more than one webcam, change the digit on `VideoCapture()` in the script to switch between devices.


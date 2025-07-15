import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from llm import ask_therapist  # Ensure llm.py exists and works

# Load emotion recognition model
model = tf.keras.models.load_model("emotion_model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detection model
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)

# Colors
primary_color = (90, 200, 250)
highlight_color = (50, 170, 200)
neutral_color = (180, 180, 180)
text_color = (20, 20, 20)
bg_color = (245, 245, 245)

# Shared state
last_emotion = None
last_response_time = 0
llm_reply = ""
waiting_for_reply = False

# User input handler
def terminal_chat_loop():
    global last_emotion, llm_reply, waiting_for_reply
    while True:
        try:
            msg = input("\nYou: ").strip()
            if not msg or not last_emotion:
                continue
            waiting_for_reply = True
            print("Thinking...")
            reply = ask_therapist(last_emotion, msg)
            print(f"Therapist: {reply}\n")
            llm_reply = reply
            waiting_for_reply = False
        except EOFError:
            break
        except Exception as e:
            print(f"(error: {e})")
            waiting_for_reply = False

# Start chat loop in separate thread
threading.Thread(target=terminal_chat_loop, daemon=True).start()

# Show message at start
show_instruction = True
start_time = time.time()

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    # Show instruction briefly
    if show_instruction and time.time() - start_time < 2:
        msg = "Press Q to Quit"
        size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(frame, msg, ((w - size[0]) // 2, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, highlight_color, 3)
    else:
        show_instruction = False

    # Process faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x2, y2 = box.astype(int)
            face = frame[y:y2, x:x2]

            if face.size > 0:
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48, 48))
                gray = gray.reshape(1, 48, 48, 1) / 255.0

                prediction = model.predict(gray, verbose=0)[0]
                emotion_idx = np.argmax(prediction)
                emotion = emotion_labels[emotion_idx]
                prob = prediction[emotion_idx]

                if emotion != last_emotion and time.time() - last_response_time > 5:
                    last_emotion = emotion
                    last_response_time = time.time()
                    print(f"[Detected Emotion]: {emotion} ({prob*100:.1f}%)")

                # Draw emotion box
                cv2.rectangle(frame, (x, y), (x2, y2), primary_color, 2)
                cv2.putText(frame, f"{emotion} ({prob*100:.1f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, primary_color, 2)

                # Emotion bar chart
                bar_x = frame.shape[1] - 170
                for j, (emo, p) in enumerate(zip(emotion_labels, prediction)):
                    bar_len = int(p * 100)
                    y_pos = 30 + j * 35

                    cv2.rectangle(frame, (bar_x - 150, y_pos),
                                  (bar_x + 100, y_pos + 25), bg_color, -1)
                    cv2.rectangle(frame, (bar_x, y_pos),
                                  (bar_x + bar_len, y_pos + 25),
                                  highlight_color if j == emotion_idx else neutral_color, -1)
                    cv2.putText(frame, f"{emo}: {int(p * 100)}%", (bar_x - 140, y_pos + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Show FPS
    fps = 1 / (time.time() - start)
    cv2.rectangle(frame, (10, 10), (125, 40), bg_color, -1)
    cv2.putText(frame, f"{fps:.1f} FPS", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    cv2.imshow("DNN Facial Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
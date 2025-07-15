import cv2
import numpy as np
import tensorflow as tf
import time

# Load emotion recognition model
model = tf.keras.models.load_model("emotion_model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector models from OpenCV
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)

show_instruction = True
start_time = time.time()

# --- Unified Color Palette ---
primary_color = (90, 200, 250)       # light blue (used for box & labels)
highlight_color = (50, 170, 200)     # darker blue for bar highlight
neutral_color = (180, 180, 180)      # soft gray
text_color = (20, 20, 20)            # almost-black for contrast
bg_color = (245, 245, 245)           # light gray-white background

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    # Show centered quit instruction (for 2 seconds)
    if show_instruction and (time.time() - start_time < 2):
        msg = "Press Q to Quit"
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, msg, (text_x, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, highlight_color, 3)
    else:
        show_instruction = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x2, y2 = box.astype(int)
            face = frame[y:y2, x:x2]

            if face.shape[0] > 0 and face.shape[1] > 0:
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (48, 48))
                gray_face = gray_face.reshape(1, 48, 48, 1) / 255.0

                prediction = model.predict(gray_face, verbose=0)[0]
                emotion_idx = np.argmax(prediction)
                emotion = emotion_labels[emotion_idx]
                prob = prediction[emotion_idx]

                # Draw emotion box
                cv2.rectangle(frame, (x, y), (x2, y2), primary_color, 2)
                cv2.putText(frame, f"{emotion} ({prob*100:.1f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, primary_color, 2)

                # Draw prediction bars
                bar_x = frame.shape[1] - 170
                for j, (emo, p) in enumerate(zip(emotion_labels, prediction)):
                    bar_length = int(p * 100)
                    y_pos = 30 + j * 35

                    # Background
                    cv2.rectangle(frame, (bar_x - 150, y_pos),
                                  (bar_x + 100, y_pos + 25),
                                  bg_color, -1)

                    # Probability bar
                    cv2.rectangle(frame, (bar_x, y_pos),
                                  (bar_x + bar_length, y_pos + 25),
                                  highlight_color if j == emotion_idx else neutral_color, -1)

                    # Label
                    cv2.putText(frame, f"{emo}: {int(p * 100)}%", (bar_x - 140, y_pos + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Draw FPS in top left
    fps = 1 / (time.time() - start)
    cv2.rectangle(frame, (10, 10), (125, 40), bg_color, -1)
    cv2.putText(frame, f"{fps:.1f} FPS", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    cv2.imshow("DNN Facial Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
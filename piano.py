import cv2
import numpy as np
import mediapipe as mp
import pygame
import os

# Khởi tạo pygame để phát âm thanh
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

# Load âm thanh và kiểm tra file
sounds = {}
sound_files = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", 
               "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6",
               "Db4", "Eb4", "Gb4", "Ab4", "Bb4",
               "Db5", "Eb5", "Gb5", "Ab5", "Bb5"]

for key in sound_files:
    path = f"sounds/{key}.wav"
    if os.path.exists(path):
        sounds[key] = pygame.mixer.Sound(path)
    else:
        print(f"⚠️ File {path} không tồn tại!")

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Định nghĩa vùng phím dưới dạng hình tứ giác
key_zones = {
    "C4": [(572, 268), (611, 268), (639, 307), (601, 307)],
    "D4": [(533, 268), (572, 268), (601, 307), (557, 307)],
    "E4": [(493, 268), (533, 268), (557, 307), (512, 307)],
    "F4": [(454, 268), (493, 268), (512, 307), (468, 307)],
    "G4": [(415, 268), (454, 268), (468, 307), (425, 307)],
    "A4": [(375, 268), (415, 268), (425, 307), (380, 307)],
    "B4": [(335, 268), (375, 268), (380, 307), (337, 307)],
    "C5": [(295, 268), (335, 268), (337, 307), (291, 307)],
    "D5": [(255, 268), (295, 268), (291, 307), (248, 307)],
    "E5": [(215, 268), (255, 268), (248, 307), (205, 307)],
    "F5": [(175, 268), (215, 268), (205, 307), (160, 307)],
    "G5": [(135, 268), (175, 268), (160, 307), (115, 307)],
    "A5": [(95, 268), (135, 268), (115, 307), (70, 307)],
    "B5": [(55, 268), (95, 268), (70, 307), (25, 307)],
    "C6": [(15, 268), (55, 268), (25, 307), (0, 307)],
}

# Mở camera
cap = cv2.VideoCapture(0)

def is_finger_pressed(finger_tip, key_zones):
    x, y = finger_tip
    for key, points in key_zones.items():
        poly = np.array(points, np.int32).reshape((-1, 1, 2))
        if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
            return key
    return None

previous_pressed_keys = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    current_pressed_keys = set()
    finger_ids = [8, 12, 16, 20]

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id in finger_ids:
                lm = hand_landmarks.landmark[id]
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                pressed_key = is_finger_pressed((cx, cy), key_zones)
                if pressed_key:
                    current_pressed_keys.add(pressed_key)
                    if pressed_key not in previous_pressed_keys and pressed_key in sounds:
                        sounds[pressed_key].play()
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    previous_pressed_keys = current_pressed_keys.copy()

    for key, points in key_zones.items():
        poly = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, key, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Piano Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

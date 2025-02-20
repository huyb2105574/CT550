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
    "C4": [(570, 228), (611, 228), (637, 250), (596, 250)],
    "D4": [(532, 228), (570, 228), (596, 250), (553, 250)],
    "E4": [(490, 228), (532, 228), (553, 250), (509, 250)],
    "F4": [(450, 228), (490, 228), (509, 250), (464, 250)],
    "G4": [(414, 228), (450, 228), (464, 250), (421, 250)],
    "A4": [(375, 228), (414, 228), (421, 250), (377, 250)],
    "B4": [(335, 228), (375, 228), (377, 250), (333, 250)],
    "C5": [(295, 228), (335, 228), (333, 250), (291, 250)],
    "D5": [(260, 228), (295, 228), (291, 250), (249, 250)],
    "E5": [(222, 228), (260, 228), (249, 250), (205, 250)],
    "F5": [(182, 228), (222, 228), (205, 250), (165, 250)],
    "G5": [(144, 228), (182, 228), (165, 250), (122, 250)],
    "A5": [(103, 228), (144, 228), (122, 250), (77, 250)],
    "B5": [(69, 228), (103, 228), (77, 250), (36, 250)],
    "C6": [(31, 228), (69, 228), (36, 250), (1, 250)],
    "Db4": [(598, 252), (619, 252), (638, 271), (637, 292)],
    "Eb4": [(548, 252), (570, 252), (617, 300), (590, 300)],
    "Gb4": [(463, 252), (490, 252), (522, 300), (494, 300)],
    "Ab4": [(411, 252), (436, 252), (461, 300), (428, 300)],
    "Bb4": [(363, 252), (388, 252), (397, 300), (370, 300)],
    "Db5": [(282, 252), (305, 252), (301, 300), (272, 300)],
    "Eb5": [(233, 252), (255, 252), (242, 300), (209, 300)],
    "Gb5": [(154, 252), (177, 252), (145, 300), (117, 300)],
    "Ab5": [(105, 252), (125, 252), (86, 300), (58, 300)],
    "Bb5": [(56, 252), (78, 252), (26, 300), (0, 300)],
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
        cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.putText(frame, key, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Piano Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp
import pygame
import os
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import time 

# Khởi tạo Flask
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Khởi tạo pygame để phát âm thanh
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

# Load âm thanh và kiểm tra file
sounds = {}
sound_files = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", 
               "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6",
            #    "Db4", "Eb4", "Gb4", "Ab4", "Bb4",
            #    "Db5", "Eb5", "Gb5", "Ab5", "Bb5"
                ]

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
    "Db4": [(598, 252), (619, 252), (638, 271), (637, 292)],
    "Eb4": [(548, 252), (570, 252), (617, 300), (590, 300)],
    "Gb4": [(463, 252), (490, 252), (522, 300), (494, 300)],
    "Ab4": [(411, 252), (436, 252), (461, 300), (428, 300)],
    "Bb4": [(363, 252), (388, 252), (397, 300), (370, 300)],
    "Db5": [(282, 252), (307, 252), (301, 300), (272, 300)],
    "Eb5": [(233, 252), (255, 252), (242, 300), (209, 300)],
    "Gb5": [(154, 252), (177, 252), (145, 300), (117, 300)],
    "Ab5": [(105, 252), (125, 252), (86, 300), (58, 300)],
    "Bb5": [(56, 252), (78, 252), (26, 300), (0, 300)],
}

# Mở camera
cap = cv2.VideoCapture(0)

def is_inside_zone(point, zone):
    poly = np.array(zone, np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0

current_song = None  # Danh sách nốt của bài hát hiện tại
current_note_index = 0  # Vị trí của nốt cần nhấn tiếp theo
pressed_keys = {}
delay = 0.5

def load_song(song_name):
    global current_song, current_note_index
    songs = {
        "Happy Birthday": ["C4", "C4", "D4", "C4", "F4", "E4"],
        "Twinkle Twinkle": ["C4", "C4", "G4", "G4", "A4", "A4", "G4"],
        "Đàn gà con": ["F4", "F4", "C4", "C4", "D4", "D4", "C4",
                        "F4", "F4", "C4", "C4", "D4", "D4", "C4",
                        "F4", "C4", "D4", "C4", "F4", "C4", "D4", "C4",
                        "C4", "C4", "D4", "E4", "F4", "F4",
                        "C4", "C4", "D4", "E4", "F4", "F4",
                        "C4", "D4", "F4", "C4", "F4", "D4", "F4", "C4"
                        ], 
        "Kìa con bướm vàng" : ["C4", "D4", "E4", "C4", "C4", "D4", "E4", "C4",
                                "E4", "F4", "G4", "E4", "F4", "G4",
                                "G4", "A4", "G4", "F4", "E4", "C4", "G4", "A4", "G4", "F4", "E4", "C4",
                                "C4", "G4", "C4", "C4", "G4", "C4"
                                ],
    }
    current_song = songs.get(song_name, [])
    current_note_index = 0
    socketio.emit('update_song', {'notes': current_song, 'current_index': current_note_index})

def generate_frames():
    global pressed_keys
    previous_pressed_keys = set()
    global current_note_index

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        current_pressed_keys = set()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for id in [8]:  # Ngón trỏ
                    lm = hand_landmarks.landmark[id]
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    for key, zone in key_zones.items():
                        if is_inside_zone((cx, cy), zone):
                            current_pressed_keys.add(key)
                            
                            # Kiểm tra nếu phím chưa từng nhấn hoặc đã qua delay giây từ lần nhấn trước
                            last_pressed_time = pressed_keys.get(key, 0)
                            current_time = time.time()

                            if key not in previous_pressed_keys and key in sounds and (current_time - last_pressed_time > delay):
                                sounds[key].play()
                                socketio.emit('note_pressed', {'note': key})  # Gửi nốt nhạc về web

                                # Cập nhật thời gian nhấn
                                pressed_keys[key] = current_time

                                # Kiểm tra nếu đúng nốt trong bài hát
                                if current_song and current_note_index < len(current_song):
                                    if key == current_song[current_note_index]:
                                        current_note_index += 1
                                        socketio.emit('update_progress', {'current_index': current_note_index})
                            break

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        previous_pressed_keys = current_pressed_keys.copy()

        for key, points in key_zones.items():
            poly = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, key, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_song/<song_name>')
def set_song(song_name):
    load_song(song_name)
    return {"status": "ok", "song": song_name}

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)

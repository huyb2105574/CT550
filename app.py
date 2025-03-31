import cv2
import numpy as np
import mediapipe as mp
import pygame
import os
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import time
import math
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///highscores.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app

app = create_app()
socketio = SocketIO(app, cors_allowed_origins="*")

class HighScore(db.Model):
    __tablename__ = "high_score"
    id = db.Column(db.Integer, primary_key=True)
    player_name = db.Column(db.String(100), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    song_name = db.Column(db.String(100), nullable=False)

# Khởi tạo pygame để phát âm thanh
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

# Load âm thanh và kiểm tra file
sounds = {}
sound_files = ["C4", "D4", "E4", "F4", "G4", "A4", "B4",
               "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6",
               "Db4", "Eb4", "Gb4", "Ab4", "Bb4",
               "Db5", "Eb5", "Gb5", "Ab5", "Bb5"
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
    "C4": [(38,72), (71, 72), (71, 214), (38, 214)],
    "D4": [(71, 72), (107, 72), (107, 214), (71, 214)],
    "E4": [(107, 72), (142, 72), (142, 214), (107, 214)],
    "F4": [(142, 72), (178, 72), (178, 214), (142, 214)],
    "G4": [(178, 72), (214, 72), (214, 214), (178, 214)],
    "A4": [(214, 72), (249, 72), (249, 214), (214, 214)],
    "B4": [(249, 72), (284, 72), (284, 214), (249, 214)],
    "C5": [(284, 72), (321, 72), (321, 214), (284, 214)],
    "D5": [(321, 72), (356, 72), (356, 214), (321, 214)],
    "E5": [(356, 72), (393, 72), (393, 214), (356, 214)],
    "F5": [(393, 72), (428, 72), (428, 214), (393, 214)],
    "G5": [(428, 72), (464, 72), (464, 214), (428, 214)],
    "A5": [(464, 72), (502, 72), (502, 214), (464, 214)],
    "B5": [(502, 72), (538, 72), (538, 214), (502, 214)],
    "C6": [(538, 72), (573, 72), (573, 214), (538, 214)],
    "Db4": [(59, 72), (79, 72), (79, 155), (59, 155)],
    "Eb4": [(100, 72), (120, 72), (120, 155), (100, 155)],
    "Gb4": [(163, 72), (183, 72), (183, 155), (163, 155)],
    "Ab4": [(204, 72), (224, 72), (224, 155), (204, 155)],
    "Bb4": [(243, 72), (263, 72), (263, 155), (243, 155)],
    "Db5": [(308, 72), (328, 72), (328, 155), (308, 155)],
    "Eb5": [(351, 72), (371, 72), (371, 155), (351, 155)],
    "Gb5": [(415, 72), (435, 72), (435, 155), (415, 155)],
    "Ab5": [(455, 72), (475, 72), (475, 155), (455, 155)],
    "Bb5": [(497, 72), (517, 72), (517, 155), (497, 155)],
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
curl_threshold = 165
min_distance = 120 # Ngưỡng góc để xác định ngón tay cong

# Hàm tính góc giữa ba điểm
def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    angle_rad = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
    return math.degrees(angle_rad)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

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
        "Kìa con bướm vàng": ["C4", "D4", "E4", "C4", "C4", "D4", "E4", "C4",
                             "E4", "F4", "G4", "E4", "F4", "G4",
                             "G4", "A4", "G4", "F4", "E4", "C4", "G4", "A4", "G4", "F4", "E4", "C4",
                             "C4", "G4", "C4", "C4", "G4", "C4"
                             ],
    }
    current_song = songs.get(song_name, [])
    current_note_index = 0
    socketio.emit('update_song', {'notes': current_song, 'current_index': current_note_index})

def generate_frames():
    global pressed_keys, black_keys
    previous_pressed_keys = set()
    global current_note_index
    black_keys = ["Db4", "Eb4", "Gb4", "Ab4", "Bb4", "Db5", "Eb5", "Gb5", "Ab5", "Bb5"]
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        current_pressed_keys = set()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, _ = frame.shape  
                landmarks = hand_landmarks.landmark
                
                # Lấy tọa độ các điểm trên ngón trỏ
                index_tip = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w,
                            landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                index_dip = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * w,
                            landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * h)
                index_pip = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * w,
                            landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * h)
                index_mcp = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * w,
                            landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * h)

                # Tính góc giữa các đốt ngón trỏ
                angle = calculate_angle(index_tip, index_dip, index_pip)
                distance = calculate_distance(index_tip, index_mcp)

                # Danh sách các phím đen (ưu tiên kiểm tra trước)
                

                key_found = None

                # Chỉ kiểm tra phím với đầu ngón trỏ
                if angle < curl_threshold or distance < min_distance:
                    for key in black_keys:
                        if is_inside_zone(index_tip, key_zones[key]):
                            key_found = key
                            break  

                    if not key_found:
                        for key in key_zones:
                            if key not in black_keys and is_inside_zone(index_tip, key_zones[key]):
                                key_found = key
                                break  

                if key_found:
                    current_pressed_keys.add(key_found)

                    last_pressed_time = pressed_keys.get(key_found, 0)
                    current_time = time.time()

                    # Chỉ phát nhạc nếu đầu ngón trỏ chạm vào và đã qua delay
                    if key_found not in previous_pressed_keys and key_found in sounds and (
                            current_time - last_pressed_time > delay):
                        sounds[key_found].play()
                        socketio.emit('note_pressed', {'note': key_found})  # Gửi nốt nhạc về web
                        pressed_keys[key_found] = current_time

                        # Kiểm tra nếu đúng nốt trong bài hát
                        if current_song and current_note_index < len(current_song):
                            if key_found == current_song[current_note_index]:
                                current_note_index += 1
                                socketio.emit('update_progress', {'current_index': current_note_index})

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        previous_pressed_keys = current_pressed_keys.copy()

        for key, points in key_zones.items():
            poly = np.array(points, np.int32).reshape((-1, 1, 2))

            # Kiểm tra nếu là phím đen, tô màu ruột
            if key in black_keys:
                cv2.fillPoly(frame, [poly], color=(0, 0, 0))  # Tô màu đen
                text_position = (points[0][0], points[0][1] - 20)  # Tên phím ở trên
                cv2.putText(frame, key, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                text_position = (points[0][0], points[0][1] + 180)  # Tên phím ở dưới
                cv2.putText(frame, key, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Vẽ viền phím
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 0), thickness=2)

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

@app.route('/save_score', methods=['POST'])
def save_score():
    data = request.get_json()
    new_score = HighScore(
        player_name=data['player_name'],
        score=data['score'],
        song_name=data['song_name']
    )
    db.session.add(new_score)
    db.session.commit()
    return jsonify({"message": "Điểm số đã được lưu!"})

# Route để lấy bảng xếp hạng theo bài hát
@app.route('/get_highscores/<song_name>', methods=['GET'])
def get_highscores(song_name):
    scores = HighScore.query.filter_by(song_name=song_name).order_by(HighScore.score.desc()).limit(10).all()
    return jsonify([
        {"player_name": score.player_name, "score": score.score}
        for score in scores
    ])

@app.route('/get_leaderboard', methods=['GET'])
def get_leaderboard():
    scores = HighScore.query.order_by(HighScore.score.desc()).limit(10).all()
    return jsonify([
        {"player_name": score.player_name, "score": score.score}
        for score in scores
    ])


if __name__ == '__main__':
    socketio.run(app, debug=True)
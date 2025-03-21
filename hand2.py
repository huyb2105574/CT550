import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Biến lưu trữ độ dài ngón tay trước đó
prev_length = None
press_threshold = 20  # Ngưỡng giảm để xác định nhấn

# Hàm tính khoảng cách Euclid giữa hai điểm
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Mở camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Lật ngược ảnh để giống gương
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Chuyển ảnh về RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ landmark bàn tay
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy tọa độ các điểm landmark của ngón trỏ
            landmarks = hand_landmarks.landmark
            index_tip = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            index_mcp = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * w,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * h)

            # Tính độ dài ngón trỏ
            finger_length = euclidean_distance(index_tip, index_mcp)

            # Kiểm tra nếu độ dài giảm đột ngột
            if prev_length is not None and prev_length - finger_length > press_threshold:
                print("🟢 Đã nhấn!")
                cv2.putText(frame, "Pressed!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # Phát âm thanh khi nhấn
                import winsound
                winsound.Beep(1000, 200)

            # Cập nhật giá trị trước đó
            prev_length = finger_length

    # Hiển thị kết quả
    cv2.imshow("Finger Press Detection", frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

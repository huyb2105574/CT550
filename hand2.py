import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Khởi tạo mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Không lật ảnh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy tọa độ TIP, PIP, MCP của ngón trỏ
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # Chuyển sang pixel
            p1 = (int(tip.x * image_width), int(tip.y * image_height))
            p2 = (int(pip.x * image_width), int(pip.y * image_height))
            p3 = (int(mcp.x * image_width), int(mcp.y * image_height))

            angle = calculate_angle(p1, p2, p3)

            # Vẽ và hiển thị độ cong
            cv2.putText(frame, f"Index Angle: {int(angle)} deg", (p2[0] + 10, p2[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Index Finger Curvature", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

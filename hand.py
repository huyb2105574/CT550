import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_finger_length(landmarks, finger_indices, image_width, image_height):
    """Tính độ dài ngón tay theo đơn vị pixel."""
    length = 0
    for i in range(len(finger_indices) - 1):
        # Chuyển đổi tọa độ chuẩn hóa về pixel
        x1, y1 = int(landmarks[finger_indices[i]].x * image_width), int(landmarks[finger_indices[i]].y * image_height)
        x2, y2 = int(landmarks[finger_indices[i + 1]].x * image_width), int(landmarks[finger_indices[i + 1]].y * image_height)
        
        # Tính khoảng cách Euclidean
        length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length

INDEX_FINGER = [5, 6, 7, 8]  # Chỉ số của ngón trỏ
PINKY_FINGER = [17, 18, 19, 20]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1)  # Lật ảnh để nhìn giống gương
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape  # Lấy kích thước ảnh
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Đo độ dài ngón trỏ theo pixel
            length_px = calculate_finger_length(hand_landmarks.landmark, INDEX_FINGER, image_width, image_height)
            cv2.putText(frame, f"Index_finger: {length_px:.2f} px", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

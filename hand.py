import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def calculate_angle_3d(a, b, c):
    """Tính góc giữa ba điểm a, b, c trong không gian 3D"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ab = a - b
    bc = c - b
    
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Đảm bảo giá trị hợp lệ
    return np.degrees(angle)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            h, w, _ = img.shape

            # Lấy tọa độ 3D của khớp ngón trỏ
            p1 = (lm[5].x * w, lm[5].y * h, lm[5].z)  # Gốc ngón trỏ
            p2 = (lm[6].x * w, lm[6].y * h, lm[6].z)  # Khớp giữa
            p3 = (lm[8].x * w, lm[8].y * h, lm[8].z)  # Đầu ngón trỏ

            # Tính góc với tọa độ 3D
            angle = calculate_angle_3d(p1, p2, p3)

            # Xác định trạng thái ngón tay dựa vào `z`
            if angle > 170:  
                status = "Straight"
            elif lm[6].z > lm[5].z and lm[6].z > lm[8].z:  # Khớp giữa bị đẩy sâu hơn gốc và đầu ngón
                status = "Bent"
            else:
                status = "Unknown"

            # Kiểm tra hướng bàn tay để tránh lỗi khi nghiêng
            wrist_z = lm[0].z  # Độ sâu của cổ tay
            finger_tip_z = lm[8].z  # Độ sâu của đầu ngón trỏ

            if abs(finger_tip_z - wrist_z) < 0.02:  # Nếu bàn tay nằm ngang
                status = "Hand Flat"

            # Vẽ landmark và hiển thị trạng thái
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, f"{status} ({int(angle)} deg)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

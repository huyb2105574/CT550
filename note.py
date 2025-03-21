import cv2
import numpy as np

def draw_piano(frame):
    height, width, _ = frame.shape
    white_width = width // 25  # Chia đều cho 25 phím trắng
    white_height = height // 2  
    black_width = int(white_width * 0.6)
    black_height = int(white_height * 0.6)

    white_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B'] * 3 + ['C', 'D', 'E', 'F']
    black_keys = ['C#', 'D#', '', 'F#', 'G#', 'A#', '', 'C#', 'D#', '', 'F#', 'G#', 'A#', '', 'C#', 'D#', '', 'F#', 'G#', 'A#', '', 'C#', 'D#', '']

    black_positions = [0.7, 1.7, None, 3.7, 4.7, 6.7, 7.7, None, 9.7, 10.7, None, 12.7, 13.7, None, 
                       15.7, 16.7, None, 18.7, 19.7, 21.7, 22.7, None, 24.7, 25.7, None]

    # Vẽ phím trắng
    for i in range(25):
        x = i * white_width
        cv2.rectangle(frame, (x, height - white_height), (x + white_width, height), (255, 255, 255), -1)
        cv2.rectangle(frame, (x, height - white_height), (x + white_width, height), (0, 0, 0), 2)
        cv2.putText(frame, white_keys[i], (x + white_width // 3, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Vẽ phím đen
    for i in range(len(black_positions)):
        if black_positions[i] is not None and black_keys[i] != '':
            x = int(black_positions[i] * white_width)
            cv2.rectangle(frame, (x, height - white_height), (x + black_width, height - white_height + black_height), (0, 0, 0), -1)
            cv2.putText(frame, black_keys[i], (x + 5, height - white_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame

# Mở webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = draw_piano(frame)

    cv2.imshow("Piano on Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

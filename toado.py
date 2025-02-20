import cv2

# Hàm callback để lấy tọa độ khi click chuột
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Khi nhấp chuột trái
        print(f"Tọa độ: ({x}, {y})")
        cv2.putText(frame, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

# Mở camera
cap = cv2.VideoCapture(0)

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", get_coordinates)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Camera", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

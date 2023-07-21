import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  
    ret, frame = cap.read()
    cv2.imwrite("frame.jpg", frame)
    cap.release()

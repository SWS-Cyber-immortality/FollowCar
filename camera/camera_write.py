import cv2

def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture the image.")
        return

    cv2.imwrite("captured_image.jpg", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Image captured successfully.")

if __name__ == "__main__":
    capture_image()

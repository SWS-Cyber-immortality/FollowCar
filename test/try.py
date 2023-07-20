import cv2
import os
import time

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2]  # Use KCF tracker

tracker = cv2.TrackerKCF_create()

video = cv2.VideoCapture(0)

ret, frame = video.read()

frame_height, frame_width = frame.shape[:2]
frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))  # Lower the resolution

if not ret:
    print('cannot read the video')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        break
    print('No face detected')
    ret, frame = video.read()
    frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))  # Lower the resolution
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imwrite('test.jpg', frame)

bbox = tuple(faces[0])
print(bbox)

Ini_frame = frame
Ini_bbox = bbox
Fail_count = 0
ret = tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))  # Lower the resolution
    if not ret:
        print('something went wrong')
        break
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        cv2.imwrite('test.jpg', frame)
    else:
        print("Tracking failure detected")
    print("FPS: ", int(fps))

video.release()
cv2.destroyAllWindows()


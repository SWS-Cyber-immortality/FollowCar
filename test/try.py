import cv2
import os
import time

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]  # Use MOSSE tracker

tracker = cv2.TrackerKCF_create()

video = cv2.VideoCapture(0)

ret, frame = video.read()

frame_height, frame_width = frame.shape[:2]
frame = cv2.resize(frame, (frame_width // 4, frame_height // 4))  # Lower the resolution
#time.sleep(5)
if not ret:
    print('cannot read the video')

# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())



# while True:
#     boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.1)  # Adjust parameters
#     if len(boxes) > 0:
#         break
#     print('No person detected')
#     ret, frame = video.read()
#     frame = cv2.resize(frame, (frame_width // 4, frame_height // 4))  # Lower the resolution

# bbox = boxes[0]

# bbox=tuple(bbox)
# print(bbox)
# this is a middle_box
# bbox = (280, 200, 100, 100)

bbox = (23, 23, 86, 320)
Ini_frame= frame
Ini_bbox = bbox
Fail_count = 0
ret = tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (frame_width // 4, frame_height // 4))  # Lower the resolution
    if not ret:
        print('something went wrong')
        break
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        print("Bounding box: ", bbox)
    else:
        print("Tracking failure detected")
    print("FPS: ", int(fps))

video.release()
cv2.destroyAllWindows()
 

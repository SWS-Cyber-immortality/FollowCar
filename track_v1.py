import cv2
import os
import sys
import numpy as np
from detect.main_yolov5 import yolov5
current_dir = os.path.dirname(os.path.abspath(__file__))

mid_anchor_x = 240
# Get the parent directory path (project_folder in this case)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to Python's sys.path
sys.path.append(parent_dir)

#from control import send_to_arduino

pre_anchor_midpoint_x = 0


def calculate_anchor_midpoint(bbox):
    # anchor_box is assumed to be a list or tuple of four values (x_min, y_min, x_max, y_max)
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    # Calculate the x-coordinate of the midpoint
    anchor_midpoint_x = int((p1[0] + p2[0]) / 2)

    return anchor_midpoint_x


# def move_motor_based_on_anchor_change(now_anchor_midpoint_x, threshold=250):
#     movement_threshold = threshold  # Set the threshold for motor movement
#
#     if abs(now_anchor_midpoint_x - mid_anchor_x) > movement_threshold:
#         if now_anchor_midpoint_x > mid_anchor_x:
#             send_to_arduino('d', '20')  # Move the motor right
#         else:
#             send_to_arduino('a', '20')  # Move the motor left
#     else:
#         send_to_arduino('w', '20')  # Move the motor forward
#
#     return now_anchor_midpoint_x

def track():
    # Start tracking
    global pre_anchor_midpoint_x
    global video
    ret, frame = video.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_180)

    #frame = cv2.resize(frame, [frame_width // 2, frame_height // 2])
    if not ret:
        print('something went wrong')
        return
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)

    now_anchor_midpoint_x = calculate_anchor_midpoint(bbox)
    #move_motor_based_on_anchor_change(now_anchor_midpoint_x)
    pre_anchor_midpoint_x = now_anchor_midpoint_x

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, "KCF Tracker", (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display the tracking result
    cv2.imshow("Tracking", frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        exit()

def detect_ini():#detect object to track and initialize the tracker
    while True:
        ret, frame = video.read()
        if not ret:
            print('cannot read the video')
            return

        # 使用yolo模型进行检测
        dets = yolonet.detect(frame)
        boxes = yolonet.postprocess(frame, dets)

        # 如果检测到目标，使用第一个检测到的目标来初始化跟踪器
        if len(boxes) > 0:
            tracker.init(frame, tuple(boxes[0]))
            break
        else:
            print("No target detected. Retrying...")




if __name__ == '__main__':
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Unable to access the camera.")
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # params = cv2.TrackerKCF_Params()
    # 设置参数
    # params.detect_thresh = 0.70
    # params.interp_factor = 0.02#
    # 使用这些参数创建跟踪器
    tracker = cv2.TrackerKCF_create()

    ret, frame = video.read()
    frame_height, frame_width = frame.shape[:2]
    yolonet = yolov5(yolo_type='yolov5s', confThreshold=0.90, nmsThreshold=0.5, objThreshold=0.5, path='./weights/')
    #detect object to track and initialize the tracker
    detect_ini()
    while True:
        track()
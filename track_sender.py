import paho.mqtt.client as mqtt
import cv2
import io
import time
import zlib  # Import the zlib library
import json
import os
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))

mid_anchor_x = 240
# Get the parent directory path (project_folder in this case)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to Python's sys.path
sys.path.append(parent_dir)

from control import send_to_arduino

pre_anchor_midpoint_x =0


def calculate_anchor_midpoint(bbox):
    # anchor_box is assumed to be a list or tuple of four values (x_min, y_min, x_max, y_max)
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    # Calculate the x-coordinate of the midpoint
    anchor_midpoint_x = int((p1[0] + p2[0]) / 2)

    return anchor_midpoint_x

def move_motor_based_on_anchor_change(now_anchor_midpoint_x, threshold=250):
    movement_threshold = threshold  # Set the threshold for motor movement

    if abs(now_anchor_midpoint_x - mid_anchor_x) > movement_threshold:
        if now_anchor_midpoint_x >  mid_anchor_x:
            send_to_arduino('d', '20')  # Move the motor right
        else:
            send_to_arduino('a', '20')  # Move the motor left
    else:
        send_to_arduino('w', '20')  # Move the motor forward
    
    return now_anchor_midpoint_x

# # Initialize the camera
# cap = cv2.VideoCapture(0)

# # # Set the camera resolution (adjust as needed)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Connect to the MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to MQTT broker")
        client.subscribe("group7/Control")
    else:
        print("Connection failed")

initial = True
def on_message(client, userdata, msg):
    global pre_anchor_midpoint_x, initial
    data = msg.payload
    dic = json.loads(data)
    # print("message received " ,str(dic))
    if dic['type'] == 'init_bbox':
        bbox = (dic['x'],dic['y'],dic['width'],dic['height'])
        global video
        ret, frame = video.read()
        ret = tracker.init(frame, bbox)
        pre_anchor_midpoint_x = calculate_anchor_midpoint(bbox)
    initial = False
    
def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client

def send_to_server(type, img):
    ret, encode_frame = cv2.imencode('.jpg', img)
    send_dict = {'type': type, 'img': encode_frame.tolist()}
    client.publish("group7/Video", json.dumps(send_dict))

def color_detection(frame):
    lower_purple = np.array([0, 0, 100])
    upper_purple = np.array([100, 100, 255])
    # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask to extract the blue color within the specified range
    mask = cv2.inRange(frame, lower_purple, upper_purple)

    # Bitwise-AND the mask with the original frame to get the color-tracked result
    result = cv2.bitwise_and(frame, frame, mask=mask)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    return result_bgr

def track():
# Start tracking
    global pre_anchor_midpoint_x
    global video
    ret, frame = video.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
   
    # frame = cv2.resize(frame, [frame_width // 2, frame_height // 2])
    frame = color_detection(frame)
    if not ret:
        print('something went wrong')
        return
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)


    now_anchor_midpoint_x = calculate_anchor_midpoint(bbox)
    move_motor_based_on_anchor_change(now_anchor_midpoint_x)
    pre_anchor_midpoint_x = now_anchor_midpoint_x

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, "MIL Tracker", (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    
    # send_to_server(type='preview',img = frame)
    print(fps)

client = setup('172.25.110.168')
if __name__ == '__main__':
   
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Unable to access the camera.")
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # tracker = cv2.TrackerMIL_create()
    # params = cv2.TrackerKCF_Params()
    # 设置参数
    # params.detect_thresh = 0.70
    # params.interp_factor = 0.02#
    # 使用这些参数创建跟踪器
    # tracker = cv2.TrackerKCF_create()
    tracker = cv2.TrackerCSRT_create()

    ret, frame = video.read()
 
    frame =cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imwrite('test.jpg', frame)
    frame = color_detection(frame)
    if not ret:
        print('cannot read the video')
    send_to_server(type='init',img =frame)
    while initial is True:
        pass
    while True:
        track()
import paho.mqtt.client as mqtt
import time
import json
import os
import sys
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory path (project_folder in this case)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to Python's sys.path
sys.path.append(parent_dir)

from control import send_to_arduino
from client.track_engine import TrackEngine

control_signal = None
action_num = 20
signal_valid = False
tracking = False
video = None
tracker = None
frame_height = None
frame_width = None

max_angle = 60#max angle of servo
mid_anchor_x = 160#mid point of anchor box
lower_bound = 90 - max_angle
upper_bound = 90 + max_angle

arduino_command = None
arduino_num = 0
def arduino_control():
    global arduino_command, arduino_num, tracking, signal_valid
    while True:
        if arduino_command is not None and (tracking is True or signal_valid is True):
            print('tracking: ', tracking, 'signal_valid: ', signal_valid, 'arduino_command: ', arduino_command, 'arduino_num: ', arduino_num)
            if arduino_num == None:
                send_to_arduino(arduino_command)
            else:
                send_to_arduino(arduino_command, str(arduino_num))
            if tracking is False and signal_valid is True and arduino_command != 'w':
                signal_valid = False
            time.sleep(0.4)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("group7/Control")
    else:
        print("Failed to connect. Error code: {}.".format(rc))

def on_message(client, userdata, msg):
    global tracking, video, tracker, frame_height, arduino_command, frame_width, signal_valid, signal_valid, control_signal, action_num
    recv_file = msg.payload
    recv_dict = json.loads(recv_file)
    print(recv_dict)
    if recv_dict['type'] == 'gesture':
        gesId = recv_dict['gesId']
        if gesId == 20:  # Thumb up: start to follow
            start_time = time.perf_counter()
            video, tracker, frame_height, frame_width = track_engine.track_prepare(0)
            end_time = time.perf_counter()
            print('track_prepare time: ', end_time - start_time)
            tracking = True
            signal_valid = False
        elif gesId == 23:  # Stop sign: stop follow, start to manual control
            control_signal='q'
            action_num = None
            tracking = False
            signal_valid = True
        elif gesId == 16 or gesId == 18:  # Zoom in: Go ahead
            control_signal = 'w'
            action_num = 20
            signal_valid = True
        elif gesId == 17 or gesId == 19:  # Zoom out: Go back
            control_signal = 's'
            action_num = 20
            signal_valid = True
        elif gesId == 0 or gesId == 6:  # Swiping left: turn left
            control_signal = 'h'
            action_num = 60
            signal_valid = True
        elif gesId == 1 or gesId == 7:  # Swiping right: turn right
            control_signal = 'g'
            action_num = 60
            signal_valid = True
        elif gesId == 18 and control_signal == 'w':
            control_signal = 'p'
            action_num = None
            signal_valid = True
        elif gesId == 19 and control_signal == 'w':
            control_signal = 'o'
            action_num = None
            signal_valid = True
        elif gesId == 24:  # purchase apple
            start_time = time.perf_counter()
            video, tracker, frame_height, frame_width = track_engine.track_prepare(47)
            end_time = time.perf_counter()
            print('track_prepare time: ', end_time - start_time)
            tracking = True
            signal_valid = False

def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client

previous_angle = None

def move_motor_based_on_anchor_change(now_anchor_midpoint_x, threshold=50):
    global previous_angle, arduino_command, arduino_num
    movement_threshold = threshold  # Set the threshold for motor movement
    gap_X = mid_anchor_x - now_anchor_midpoint_x  # Change the order to invert the direction

    # Map the x-coordinate of the target (now_anchor_midpoint_x) from the image coordinate system (0 to 320)
    # to the servo coordinate system (lower_bound to upper_bound)
    angle = -0.1875*now_anchor_midpoint_x + 120
    # angle = upper_bound + ((now_anchor_midpoint_x / 320) * (upper_bound - lower_bound))
    
    # print("angle: ", angle)
    # print("previous_angle: ", previous_angle)
    # print("x: ", now_anchor_midpoint_x)
    # Only send the new angle to the Arduino if the difference with the previous angle is larger than 10 degrees
    if previous_angle is None or abs(previous_angle - angle) > 2:
        send_to_arduino('r', str(int(angle)))  # Move the servo
        previous_angle = angle

    if abs(gap_X) > movement_threshold:
        if now_anchor_midpoint_x > mid_anchor_x:
            arduino_command = 'd'
            arduino_num = 20
        else:
            arduino_command = 'a'
            arduino_num = 20
    else:
        arduino_command = 'w'
        arduino_num = 10


if __name__ == '__main__':
    track_engine = TrackEngine()
    client = setup('192.168.43.41')
   

    arduino_signal_thread = threading.Thread(target=arduino_control)
    arduino_signal_thread.daemon = True
    arduino_signal_thread.start()

    while True:
        if tracking is True:
            now_anchor_midpoint_x = track_engine.track(tracker, video, frame_height, frame_width)
            move_motor_based_on_anchor_change(now_anchor_midpoint_x)
        elif signal_valid is True:
            arduino_command = control_signal
            arduino_num = action_num
            track_engine.preview_from_camera()
        else:
            track_engine.preview_from_camera()

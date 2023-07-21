import paho.mqtt.client as mqtt
import time
import json
from control import send_to_arduino
from client.track_engine import *


control_signal = None
action_num = 20
signal_valid = False
tracking = False
video = None
tracker = None
frame_height = None
frame_width = None

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Control")
    else:
        print("Failed to connect. Error code: {}.".format(rc))

def on_message(client, userdata, msg):
    global tracking, video, tracker, frame_height, frame_width, signal_valid, signal_valid, control_signal, action_num
    recv_file = msg.payload
    recv_dict = json.loads(recv_file)
    if recv_dict['type'] == 'gesture':
        gesId = recv_dict['gesId']
        if gesId == 20:  # Thumb up: start to follow
            video, tracker, frame_height, frame_width = track_prepare()
            tracking = True
            signal_valid = False
        elif gesId == 21:  # Thumb down: Go ahead
            control_signal = 'w'
            action_num = 20
            signal_valid = True
        elif gesId == 23:  # Stop sign: stop follow, start to manual control
            tracking = False
        elif gesId == 0 or gesId == 6:  # Swiping left: turn left
            control_signal = 'a'
            action_num = 90
            signal_valid = False
        elif gesId == 1 or gesId == 7:  # Swiping right: turn right
            control_signal = 'd'
            action_num = 90
            signal_valid = False

def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client

def move_motor_based_on_anchor_change(now_anchor_midpoint_x, threshold=250):
    movement_threshold = threshold  # Set the threshold for motor movement

    if abs(now_anchor_midpoint_x - mid_anchor_x) > movement_threshold:
        if now_anchor_midpoint_x > mid_anchor_x:
            send_to_arduino('d', '20')  # Move the motor right
        else:
            send_to_arduino('a', '20')  # Move the motor left
    else:
        send_to_arduino('w', '20')  # Move the motor forward

if __name__ == '__main__':
    client = setup('172.25.99.30')
    while True:
        if tracking is True:
            now_anchor_midpoint_x = track(tracker, video, frame_height, frame_width)
            move_motor_based_on_anchor_change(now_anchor_midpoint_x)
            time.sleep(0.05)
        elif signal_valid is True:
            if control_signal == 'w':
                send_to_arduino('w', '20')
            else:
                send_to_arduino(control_signal, str(action_num))
                signal_valid = False
            time.sleep(0.1)

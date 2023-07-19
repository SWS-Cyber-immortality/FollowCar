import paho.mqtt.client as mqtt
import cv2
import numpy as np

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Video")
    else:
        print("Failed to connect. Error code: {}.".format(rc))

message_cnt = 0
def on_message(client, userdata, msg):
    global message_cnt
    message_cnt += 1
    print("Receive message{}".format(message_cnt))
    recv_file = msg.payload
    img = np.frombuffer(recv_file, np.uint8).reshape((640,640,3))
    cv2.imshow('Image Stream', img)
    cv2.waitKey(1)

def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client

if __name__ == '__main__':
    client = setup('172.31.70.119')
    while True:
        pass
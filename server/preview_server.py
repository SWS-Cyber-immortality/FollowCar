import paho.mqtt.client as mqtt
import cv2
import numpy as np
import time

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Video")
    else:
        print("Failed to connect. Error code: {}.".format(rc))

message_cnt = 0

last_time = time.perf_counter()
def on_message(client, userdata, msg):
    global message_cnt
    global last_time
    message_cnt += 1
    print("Receive message{}".format(message_cnt))
    if message_cnt > 1:
        cost_time = time.perf_counter() - last_time
        print('cost time:{}s'.format(cost_time))
    last_time = time.perf_counter()
    recv_file = msg.payload
    img = np.frombuffer(recv_file, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
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
    client = setup('172.25.99.30')
    while True:
        pass
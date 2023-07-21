import paho.mqtt.client as mqtt
import cv2
import numpy as np
import time
import json

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("group7/Video")
    else:
        print("Failed to connect. Error code: {}.".format(rc))

def send_dict(msgtype, data):
    dic = None
    if msgtype == 'init_bbox':
        dic = {
            'type': msgtype,
            'x': data[0],
            'y': data[1],
            'width': data[2],
            'height': data[3]
        }
    if dic is not None:
        print('send dict:', dic)
        msg = json.dumps(dic)
        client.publish("group7/Control", msg)

def on_message(client, userdata, msg):
    data = msg.payload
    data_dict = json.loads(data)
    print("Receive {} dict message".format(data_dict['type']))
    # print(data_dict)
    if data_dict['type'] == 'init':
        img = data_dict['img']
        img = np.array(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        bbox = cv2.selectROI(img, False)
        print('bbox',bbox)
        send_dict('init_bbox',bbox)
    if data_dict['type'] == 'preview':
        img = data_dict['img']
        img = np.array(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=2.0, fy=2.0)
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
    client = setup('172.25.110.168')
    while True:
        pass
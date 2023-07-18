import paho.mqtt.client as mqtt
import time
import cv2

client = mqtt.Client()
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Video")
    else:
        print("Failed to connect. Error code: {}.".format(rc))

starttime = time.perf_counter()

def send_image():
#   starttime = time.perf_counter()
    img = cv2.imread("captured_image.jpg")
    starttime = time.perf_counter()
    client.publish("Video", img.tobytes())
    


def on_message(client, userdata, msg):
    global starttime
    endtime = time.perf_counter()
    print("Receive result:{}s".format(endtime-starttime))
    

def setup(hostname):
 
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client

if __name__ == '__main__':
    setup('172.25.101.155')
    send_image()
    while True:
        pass
 

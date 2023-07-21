import paho.mqtt.client as mqtt
import cv2
import io
import time
import zlib  # Import the zlib library
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

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

def move_motor_based_on_anchor_change(now_anchor_midpoint_x, pre_anchor_midpoint_x, threshold=30):
    movement_threshold = threshold  # Set the threshold for motor movement

    if abs(now_anchor_midpoint_x - pre_anchor_midpoint_x) > movement_threshold:
        if now_anchor_midpoint_x > pre_anchor_midpoint_x:
            send_to_arduino('d', '20')  # Move the motor right
        else:
            send_to_arduino('a', '20')  # Move the motor left
    else:
        send_to_arduino('w', '20')  # Move the motor forward
    
    return now_anchor_midpoint_x

# MQTT broker configuration
broker_address = "172.25.110.168"  # Update with your broker address
broker_port = 1883  # Update with your broker port
topic = "group7/Video"  # Update with your desired topic

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
start_time = time.perf_counter()
def on_message(client, userdata, msg):
    end_time = time.perf_counter()
    print("Receive  result:{}s".format(end_time-start_time))
    global pre_anchor_midpoint_x, initial
    data = msg.payload
    dic = json.loads(data)
    if dic['type'] == 'init_bbox':
        bbox =(dic['x'],dic['y'],dic['width'],dic['height']) 
        frame = cv2.imread("../picture/first.jpg")
        ret = tracker.init(frame, bbox)
        pre_anchor_midpoint_x = calculate_anchor_midpoint(bbox)
    initial = False
    

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker_address, broker_port, 60)
client.loop_start()





now = time.perf_counter()

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[5]
tracker = cv2.TrackerMIL_create() #Nano,GOTURN, MIL, DaSiamRPN

def timeit(func):
    def wrapper(*args, **kwargs):
      
        result = func(*args, **kwargs)
        end = time.perf_counter() - start_time
        print(func.__name__,end)
        return result
    return wrapper



def send_to_server(type,img):
    # Capture and publish video frames continuousl
    # Publish the compressed frame to the MQTT broker
       # print(type(img))
       
       #  cv2.imwrite("output.jpg",encoded_frame)
        # Publish the compressed frame to the MQTT broker
        ret, encode_frame = cv2.imencode('.jpg',img)
        send_dict = {'type':type,'img': encode_frame.tolist()}

        # if not ret:
        #     print("fail to encode")
        #   print(type(encode_frame))
        print("send",topic)
        client.publish(topic, json.dumps(send_dict))
        # time.sleep(0.1)

send_to_server(type='init',img = 1)   
while True:
    pass 
    # finally:
    # #  Clean up resources
    #     client.loop_stop()
    #     client.disconnect()
    # #    cap.release()
    #     cv2.destroyAllWindows()
def track():
# Start tracking
    global pre_anchor_midpoint_x
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    ret, frame = video.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # frame = cv2.resize(frame, [frame_width // 2, frame_height // 2])
    if not ret:
        print('something went wrong')
        return
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)

    now_anchor_midpoint_x = calculate_anchor_midpoint(bbox)
    move_motor_based_on_anchor_change(now_anchor_midpoint_x, pre_anchor_midpoint_x)
    pre_anchor_midpoint_x = now_anchor_midpoint_x

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, tracker_type + " Tracker", (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    
    send_to_server(type='preview',img = frame)
    # print(fps)
    # output.write(frame)
    # k = cv2.waitKey(1) & 0xff
    video.release()

  

def test():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        # Read a frame from the camera
        ret, srcimg = cap.read()
        if not ret:
            break
        send_to_server(srcimg)
# test()
import paho.mqtt.client as mqtt
import cv2
import numpy as np
import time
import json
import torch
from torchvision.transforms import *
import torchvision.transforms.transforms as t
from gesture_recognition.FullModel import FullModel
from PIL import Image
import pandas as pd

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
    # client.publish("group7/Control", data)
    print("Receive {} message".format(data_dict['type']))
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

label_dict = pd.read_csv('../gesture_recognition/jester-v1-labels.csv', header=None)
ges = label_dict[0].tolist()
mode = 0 # 0 for initial, 1 for following, 2 for manual
def handle_gesture(indice):
    global mode
    print(ges[indice])
    valid = True
    action = ''
    if indice == 20 and mode != 1:  # Thumb up: start to follow
        action = 'follow'
        mode = 1
    elif indice == 21 and mode == 2:  # Thumb down: Go ahead
        action = 'go'
    elif indice == 23 and mode != 2:  # Stop sign: stop follow, start to manual control
        action = 'stop'
        mode = 2
    elif indice == 0 and mode == 2:  # Swiping left: turn left
        action = 'left'
    elif indice == 1 and mode == 2:  # Swiping right: turn right
        action = 'right'
    # elif indice == 18 and mode == 2:  # Zooming In With Two Fingers: accelerate
    #     action = 'accelerate'
    # elif indice == 19 and mode == 2: # Zooming Out With Two Fingers: slow down
    #     action = 'slow'
    else:
        valid = False

    if valid:
        send_dict('gesture', (indice, action))

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 60)
    # Set up some storage variables
    seq_len = 16
    value = 0
    imgs = []
    pred = 8
    top_3 = [9, 8, 7]
    out = np.zeros(10)

    # Load model
    print('Loading model...')
    curr_folder = 'models_jester'
    model = FullModel(batch_size=1, seq_lenght=16)
    loaded_dict = torch.load('../gesture_recognition/gr.ckp')
    model.load_state_dict(loaded_dict)
    model = model.cuda()
    model.eval()

    std, mean = [0.2674, 0.2676, 0.2648], [0.4377, 0.4047, 0.3925]
    transform = Compose([
        t.CenterCrop((96, 96)),
        t.ToTensor(),
        t.Normalize(std=std, mean=mean),
    ])

    print('Gesture Recognition Ready')

    s = time.time()
    n = 0
    hist = []
    mean_hist = []
    cooldown = 0
    eval_samples = 2
    num_classes = 27

    score_energy = torch.zeros((eval_samples, num_classes))

    client = setup('172.25.110.168')
    while True:
        ret, frame = cam.read()
        # Set up input for model
        resized_frame = cv2.resize(frame, (160, 120))
        pre_img = Image.fromarray(resized_frame.astype('uint8'), 'RGB')

        img = transform(pre_img)

        if n % 4 == 0:
            imgs.append(torch.unsqueeze(img, 0))

        # Get model output prediction
        if len(imgs) == 16:
            data = torch.cat(imgs).cuda()
            output = model(data.unsqueeze(0))
            out = (torch.nn.Softmax()(output).data).cpu().numpy()[0]
            if len(hist) > 300:
                mean_hist = mean_hist[1:]
                hist = hist[1:]
            out[-2:] = [0, 0]
            hist.append(out)
            score_energy = torch.tensor(hist[-eval_samples:])
            curr_mean = torch.mean(score_energy, dim=0)
            mean_hist.append(curr_mean.cpu().numpy())
            # value, indice = torch.topk(torch.from_numpy(out), k=1)
            value, indice = torch.topk(curr_mean, k=1)
            indices = np.argmax(out)
            top_3 = out.argsort()[-3:]
            if cooldown > 0:
                cooldown = cooldown - 1
            if value.item() > 0.6 and indices < 25 and cooldown == 0:
                print('Gesture:', ges[indices], '\t\t\t\t\t\t Value: {:.2f}'.format(value.item()))
                handle_gesture(indice)
                cooldown = 16
            pred = indices
            imgs = imgs[1:]

        n += 1
        bg = np.full((480, 1200, 3), 15, np.uint8)
        bg[:480, :640] = frame

        font = cv2.FONT_HERSHEY_SIMPLEX
        if value > 0.6:
            cv2.putText(bg, ges[pred], (40, 40), font, 1, (0, 0, 0), 2)
        cv2.rectangle(bg, (128, 48), (640 - 128, 480 - 48), (0, 255, 0), 3)
        for i, top in enumerate(top_3):
            cv2.putText(bg, ges[top], (700, 200 - 70 * i), font, 1, (255, 255, 255), 1)
            cv2.rectangle(bg, (700, 225 - 70 * i), (int(700 + out[top] * 170), 205 - 70 * i), (255, 255, 255), 3)
        cv2.imshow('gesture', bg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
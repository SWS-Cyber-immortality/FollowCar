import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import *
import torchvision.transforms.transforms as t
import time
import warnings
from FullModel import FullModel
warnings.filterwarnings('ignore')

label_dict = pd.read_csv('./jester-v1-labels.csv', header=None)
ges = label_dict[0].tolist()
for g in ges:
	print(g)

if __name__ == '__main__':
	# Capture video from computer camera
	cam = cv2.VideoCapture(0)
	cam.set(cv2.CAP_PROP_FPS, 60)

	# Set up some storage variables
	seq_len = 16
	value = 0
	imgs = []
	pred = 8
	top_3 = [9,8,7]
	out = np.zeros(10)
	# Load model
	print('Loading model...')

	curr_folder = 'models_jester'
	model = FullModel(batch_size=1, seq_lenght=16)
	loaded_dict = torch.load('./gr.ckp')
	model.load_state_dict(loaded_dict)
	model = model.cuda()
	model.eval()

	std, mean = [0.2674,  0.2676,  0.2648], [ 0.4377,  0.4047,  0.3925]
	transform = Compose([
		t.CenterCrop((96, 96)),
		t.ToTensor(),
		t.Normalize(std=std, mean=mean),
	])

	print('Starting prediction')

	n = 0
	hist = []
	mean_hist = []
	cooldown = 0
	eval_samples = 2
	num_classes = 27

	score_energy = torch.zeros((eval_samples, num_classes))

	while(True):
		# Capture frame-by-frame
		ret, frame = cam.read()
		# Set up input for model
		resized_frame = cv2.resize(frame, (160, 120))
		pre_img = Image.fromarray(resized_frame.astype('uint8'), 'RGB')

		img = transform(pre_img)

		if n%4 == 0:
			imgs.append(torch.unsqueeze(img, 0))

		# Get model output prediction
		if len(imgs) == 16:
			data = torch.cat(imgs).cuda()
			output = model(data.unsqueeze(0))
			out = (torch.nn.Softmax()(output).data).cpu().numpy()[0]
			if len(hist) > 300:
				mean_hist  = mean_hist[1:]
				hist  = hist[1:]
			out[-2:] = [0,0]
			hist.append(out)
			score_energy = torch.tensor(hist[-eval_samples:])
			curr_mean = torch.mean(score_energy, dim=0)
			mean_hist.append(curr_mean.cpu().numpy())
			#value, indice = torch.topk(torch.from_numpy(out), k=1)
			value, indice = torch.topk(curr_mean, k=1)
			indices = np.argmax(out)
			top_3 = out.argsort()[-3:]
			if cooldown > 0:
				cooldown = cooldown - 1
			if value.item() > 0.5 and indices < 25 and cooldown == 0:
				print('Gesture:', ges[indices], '\t\t\t\t\t\t Value: {:.2f}'.format(value.item()))
				cooldown = 16
			pred = indices
			imgs = imgs[1:]

		n += 1
		bg = np.full((480, 1200, 3), 15, np.uint8)
		bg[:480, :640] = frame

		font = cv2.FONT_HERSHEY_SIMPLEX
		if value > 0.6:
			cv2.putText(bg, ges[pred],(40,40), font, 1,(0,0,0),2)
		cv2.rectangle(bg,(128,48),(640-128,480-48),(0,255,0),3)
		for i, top in enumerate(top_3):
			cv2.putText(bg, ges[top],(700,200-70*i), font, 1,(255,255,255),1)
			cv2.rectangle(bg,(700,225-70*i),(int(700+out[top]*170),205-70*i),(255,255,255),3)

		cv2.imshow('gesture',bg)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cam.release()
	cv2.destroyAllWindows()
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import *
import torchvision.transforms.transforms as t
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import time
import warnings
warnings.filterwarnings('ignore')

class FullModel(nn.Module):

	def __init__(self, batch_size, seq_lenght=8):
		super(FullModel, self).__init__()

		class CNN2D(nn.Module):
			def __init__(self, batch_size=batch_size, image_size=96, seq_lenght=8, in_channels=3):
				super(CNN2D, self).__init__()
				self.conv1 = self._create_conv_layer(in_channels=in_channels, out_channels=16)
				self.conv2 = self._create_conv_layer(in_channels=16, out_channels=32)
				self.conv3 = self._create_conv_layer_pool(in_channels=32, out_channels=64)
				self.conv4 = self._create_conv_layer_pool(in_channels=64, out_channels=128)
				self.conv5 = self._create_conv_layer_pool(in_channels=128, out_channels=256)
				cnn_output_shape = int(256 * (image_size / (2 ** 4)) ** 2)

			def forward(self, x):
				batch_size, frames, channels, width, height = x.shape
				x = x.view(-1, channels, width, height)
				x = self.conv1(x)
				x = self.conv2(x)
				x = self.conv3(x)
				x = self.conv4(x)
				x = self.conv5(x)
				return x

			def _create_conv_layer(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
				return nn.Sequential(
					nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
					nn.BatchNorm2d(out_channels),
					nn.ReLU(),
				)

			def _create_conv_layer_pool(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1),
										pool=(2, 2)):
				return nn.Sequential(
					nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
					nn.BatchNorm2d(out_channels),
					nn.ReLU(),
					nn.MaxPool2d(pool)
				)

		class CNN3D(nn.Module):
			def __init__(self, batch_size=batch_size, image_size=96, seq_lenght=8):
				super(CNN3D, self).__init__()
				self.conv1 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(1, 1, 1))
				self.conv2 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(2, 2, 2))
				self.conv3 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(2, 1, 1))
				self.conv4 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(2, 2, 2))

			def forward(self, x):
				batch_size, channels, frames, width, height = x.shape
				x = self.conv1(x)
				x = self.conv2(x)
				x = self.conv3(x)
				x = self.conv4(x)
				return x

			def _create_conv_layer(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
				return nn.Sequential(
					nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
					nn.BatchNorm3d(out_channels),
					nn.ReLU(),
				)

			def _create_conv_layer_pool(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1),
										pool=(1, 2, 2)):
				return nn.Sequential(
					nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
					nn.BatchNorm3d(out_channels),
					nn.ReLU(),
					nn.MaxPool3d(pool)
				)

		class Combiner(nn.Module):

			def __init__(self, in_features):
				super(Combiner, self).__init__()
				self.linear1 = self._create_linear_layer(in_features, in_features // 2)
				self.linear2 = self._create_linear_layer(in_features // 2, 1024)
				self.linear3 = self._create_linear_layer(1024, 27)

			def forward(self, x):
				x = self.linear1(x)
				x = self.linear2(x)
				x = self.linear3(x)
				return x

			def _create_linear_layer(self, in_features, out_features, p=0.6):
				return nn.Sequential(
					nn.Linear(in_features, out_features),
					nn.Dropout(p=p)
				)

		self.rgb2d = CNN2D(batch_size)
		self.rgb3d = CNN3D(batch_size)
		self.combiner = Combiner(4608)

		self.batch_size = batch_size
		self.seq_lenght = seq_lenght
		self.steps = 0
		self.steps = 0
		self.epochs = 0
		self.best_valdiation_loss = math.inf

	def forward(self, x):
		self.batch_size = x.shape[0]
		x = self.rgb2d(x)
		batch_and_frames, channels, dim1, dim2 = x.shape
		x = x.view(self.batch_size, -1, channels, dim1, dim2).permute(0, 2, 1, 3, 4)
		x = self.rgb3d(x)
		x = x.view(self.batch_size, -1)
		x = self.combiner(x)

		if self.training:
			self.steps += 1

		return x

label_dict = pd.read_csv('./jester-v1-labels.csv', header=None)
ges = label_dict[0].tolist()
# print(ges)

# Capture video from computer camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 30)

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

s = time.time()
n = 0
hist = []
mean_hist = []
setup = True
plt.ion()
fig, ax = plt.subplots()
cooldown = 0
eval_samples = 2
num_classes = 27

score_energy = torch.zeros((eval_samples, num_classes))

while(True):
	# Capture frame-by-frame
	ret, frame = cam.read()
	#print(np.shape(frame)) # (480, 640, 3)
	# Set up input for model
	resized_frame = cv2.resize(frame, (160, 120))

	#print(np.shape(resized_frame))

	pre_img = Image.fromarray(resized_frame.astype('uint8'), 'RGB')

	#print(np.shape(pre_img))

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
		if value.item() > 0.6 and indices < 25 and cooldown == 0:
			print('Gesture:', ges[indices], '\t\t\t\t\t\t Value: {:.2f}'.format(value.item()))
			cooldown = 16
		pred = indices
		imgs = imgs[1:]

		df=pd.DataFrame(mean_hist, columns=ges)

		ax.clear()
		df.plot.line(legend=False, figsize=(16,6),ax=ax, ylim=(0,1))
		if setup:
			plt.show(block = False)
			setup=False
		plt.draw()

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

	cv2.imshow('preview',bg)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
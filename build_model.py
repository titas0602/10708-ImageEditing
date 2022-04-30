import numpy as np
import torch
import kornia
from torch import nn
import torch.nn.functional as F
#from Transform import rotate, rotate_pad_zero, rotate_pad_mean, translate3d_zyz
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
from torch.autograd import Variable
from functional import reset_normal_param, LinearWeightNorm

def discriminator(self):        
	D = Sequential()
	
	#add Gaussian noise to prevent Discriminator overfitting
	D.add(GaussianNoise(0.2, input_shape = [256, 256, 3]))
	
	#256x256x3 Image
	D.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same'))
	D.add(LeakyReLU(0.2))
	D.add(Dropout(0.25))
	D.add(AveragePooling2D())
	
	#128x128x8
	D.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same'))
	D.add(BatchNormalization(momentum = 0.7))
	D.add(LeakyReLU(0.2))
	D.add(Dropout(0.25))
	D.add(AveragePooling2D())
	
	#64x64x16
	D.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same'))
	D.add(BatchNormalization(momentum = 0.7))
	D.add(LeakyReLU(0.2))
	D.add(Dropout(0.25))
	D.add(AveragePooling2D())
	
	#32x32x32
	D.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same'))
	D.add(BatchNormalization(momentum = 0.7))
	D.add(LeakyReLU(0.2))
	D.add(Dropout(0.25))
	D.add(AveragePooling2D())
	
	#16x16x64
	D.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same'))
	D.add(BatchNormalization(momentum = 0.7))
	D.add(LeakyReLU(0.2))
	D.add(Dropout(0.25))
	D.add(AveragePooling2D())
	
	#8x8x128
	D.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same'))
	D.add(BatchNormalization(momentum = 0.7))
	D.add(LeakyReLU(0.2))
	D.add(Dropout(0.25))
	self.D.add(AveragePooling2D())
	
	#4x4x256
	D.add(Flatten())
	
	#256
	D.add(Dense(128))
	D.add(LeakyReLU(0.2))
	
	D.add(Dense(1))
	
	return self.D


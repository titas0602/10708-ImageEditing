import numpy as np
import torch
# import kornia
from torch import nn
import torch.nn.functional as F
#from Transform import rotate, rotate_pad_zero, rotate_pad_mean, translate3d_zyz
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
from torch.autograd import Variable
# from functional import reset_normal_param, LinearWeightNorm
import torch.nn.init as init

class UpSampleConv2D(nn.Module):
	# TODO 1.1: Implement nearest neighbor upsampling + conv layer

	def __init__(
		self,
		input_channels,
		kernel_size=3,
		n_filters=128,
		upscale_factor=2,
		padding=0,
	):
		super(UpSampleConv2D, self).__init__()
		self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=padding)
		self.upscale_factor = upscale_factor
		self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

	def forward(self, x):
		# TODO 1.1: Implement nearest neighbor upsampling
		# 1. Stack x channel wise upscale_factor^2 times
		# 2. Then re-arrange to form a batch x channel x height*upscale_factor x width*upscale_factor
		# 3. Apply convolution.
		# Hint for 2. look at
		# https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle
		# x = x.repeat(1, self.upscale_factor*self.upscale_factor, 1, 1)
		# x = self.pixel_shuffle(x)
		# x = x.repeat(1, self.upscale_factor*self.upscale_factor, 1, 1)
		# print(x.size())
		x = x.repeat_interleave(self.upscale_factor, dim=2)
		x = x.repeat_interleave(self.upscale_factor, dim=3)
		# print(x.size())
		x = self.conv(x)
		return x


class DownSampleConv2D(nn.Module):
	# TODO 1.1: Implement spatial mean pooling + conv layer

	def __init__(
		self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
	):
		super(DownSampleConv2D, self).__init__()
		self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=padding)
		self.downscale_ratio = downscale_ratio
		self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_ratio)
		self.input_channels = input_channels

	def forward(self, x):
		# TODO 1.1: Implement spatial mean pooling
		# 1. Re-arrange to form a batch x channel * upscale_factor^2 x height x width
		# 2. Then split channel wise into batch x channel x height x width Images
		# 3. average the images into one and apply convolution
		# Hint for 1. look at
		# https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle
		x = self.pixel_unshuffle(x)
		split = torch.split(x, self.input_channels, dim=1)
		split = list(split)

		# x = torch.stack(split)
		for i in range(0, len(split)):
			split[i] = split[i].unsqueeze(0)
		x = torch.cat(split, dim = 0)
		#print(x.size())
		x = torch.transpose(x, 0, 1)
		#print(x.size())
		x = torch.mean(x, dim=1)
		#print(x.size())

		x = self.conv(x)
		# print("Down")
		# print(x.size())
		return x


class ResBlockUp(nn.Module):
	# TODO 1.1: Impement Residual Block Upsampler.
	"""
	ResBlockUp(
		(layers): Sequential(
			(0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(1): ReLU()
			(2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			(3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(4): ReLU()
		)
		(residual): UpSampleConv2D(
			(conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		)
		(shortcut): UpSampleConv2D(
			(conv): Conv2d(n_filters, n_filters, kernel_size=(1, 1), stride=(1, 1))
		)
	"""

	def __init__(self, input_channels, kernel_size=3, n_filters=128, leak=False):
		super(ResBlockUp, self).__init__()
		if not leak:
			self.layers = nn.Sequential(nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(), nn.Conv2d(input_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU())
		else:
			self.layers = nn.Sequential(nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.LeakyReLU(0.2), nn.Conv2d(input_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.LeakyReLU(0.2))
		self.residual = UpSampleConv2D(n_filters, n_filters=n_filters, padding=1)
		self.shortcut = UpSampleConv2D(input_channels, n_filters=n_filters, kernel_size=1)

	def forward(self, x):
		# TODO 1.1: Forward through the layers and implement a residual connection.
		# Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
		out_layers = self.layers(x)
		out_layers = self.residual(out_layers)
		out_shortcut = self.shortcut(x)

		return out_layers+out_shortcut



class ResBlockDown(nn.Module):
	# TODO 1.1: Impement Residual Block Downsampler.
	'''
	ResBlockDown(
		(layers): Sequential(
		(0): ReLU()
		(1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		 (2): ReLU()
		 )
		(residual): DownSampleConv2D(
		(conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		 )
		 (shortcut): DownSampleConv2D(
		 (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
		 )
		 )
	'''

	def __init__(self, input_channels, kernel_size=3, n_filters=128, leak=False):
		super(ResBlockDown, self).__init__()
		if not leak:
			self.layers = nn.Sequential(nn.ReLU(), nn.Conv2d(input_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU())
		else:
			self.layers = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(input_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.LeakyReLU(0.2))
		self.residual = DownSampleConv2D(n_filters, n_filters=n_filters, padding=1)
		self.shortcut = DownSampleConv2D(input_channels, n_filters=n_filters, kernel_size=1)

	def forward(self, x):
		# TODO 1.1: Forward through the layers and implement a residual connection.
		# Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
		out_layers = self.layers(x)
		out_layers = self.residual(out_layers)
		out_shortcut = self.shortcut(x)

		return out_layers+out_shortcut


class ResBlock(nn.Module):
	# TODO 1.1: Impement Residual Block as described below.
	"""
	ResBlock(
		(layers): Sequential(
			(0): ReLU()
			(1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			(2): ReLU()
			(3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		)
	)
	"""

	def __init__(self, input_channels, kernel_size=3, n_filters=128, leak=False):
		super(ResBlock, self).__init__()
		if not leak:
			self.layers = nn.Sequential(nn.ReLU(), nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1)), nn.ReLU(), nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1)))
		else:
			self.layers = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1)), nn.LeakyReLU(0.2), nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1)))


	def forward(self, x):
		# TODO 1.1: Forward the conv layers. Don't forget the residual connection!
		# print(x.size())
		return self.layers(x) + x

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
		if hasattr(m, "bias") and m.bias is not None:
			torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
	def __init__(self, in_features):
		super(ResidualBlock, self).__init__()

		self.block = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(in_features, in_features, 3),
			nn.InstanceNorm2d(in_features),
			nn.ReLU(inplace=True),
			nn.ReflectionPad2d(1),
			nn.Conv2d(in_features, in_features, 3),
			nn.InstanceNorm2d(in_features),
		)

	def forward(self, x):
		return x + self.block(x)


class GeneratorResNet(nn.Module):
	def __init__(self, input_shape, num_residual_blocks):
		super(GeneratorResNet, self).__init__()

		channels = input_shape[0]

		# Initial convolution block
		out_features = 128
		model = [
			#nn.ReflectionPad2d(channels),
			nn.Conv2d(channels, out_features, 3, stride=1, padding=1),
			nn.InstanceNorm2d(out_features),
			nn.ReLU(inplace=True),
		]
		in_features = out_features

		"""
		# Downsampling
		for _ in range(2):
			out_features *= 2
			model += [
				nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
				nn.InstanceNorm2d(out_features),
				nn.ReLU(inplace=True),
			]
			in_features = out_features
		"""

		# Residual blocks
		for _ in range(num_residual_blocks):
			model += [ResidualBlock(out_features)]

		# Upsampling
		for _ in range(2):
			out_features //= 2
			model += [
				nn.Upsample(scale_factor=2),
				nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
				nn.InstanceNorm2d(out_features),
				nn.ReLU(inplace=True),
			]
			in_features = out_features

		# Output layer
		model += [nn.Conv2d(32, 3, 3, stride=1, padding=1), nn.Tanh()]

		self.model = nn.Sequential(*model)
		self.channels = input_shape[0]
		self.fc = nn.Linear(input_shape[0], input_shape[0]*16*16)

	def forward(self, x):
		x = self.fc(x)
		x = nn.ReLU()(x)
		x = torch.reshape(x, (-1, self.channels, 16, 16))
		return self.model(x)

class Discriminator(nn.Module):
	def __init__(self, input_shape):
		super(Discriminator, self).__init__()

		self.conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.LeakyReLU(0.1), nn.BatchNorm2d(32), nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.LeakyReLU(0.1),  nn.BatchNorm2d(64), nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.LeakyReLU(0.1), nn.BatchNorm2d(128), nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.LeakyReLU(0.1))
		self.bn1 = nn.BatchNorm1d(4*4*512)
		self.fc1 = nn.Linear(8*8*1, 1)
		self.bn2 = nn.BatchNorm1d(100)
		self.fc2 = nn.Linear(100, 1)

	def forward(self, img):
		cl = self.conv(img)
		cl = nn.Flatten()(cl)
		cl = self.fc1(cl)
		#print(img.size())
		#img = nn.Flatten()(img)
		#img = self.bn1(img)
		#img = nn.ReLU()(img)
		#img = self.fc1(img)
		#img = self.bn2(img)
		#img = nn.ReLU()(img)
		#cl = self.fc2(img)
		# print(out.size())
		return cl

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance

        x = torch.cat([x, o_b], 1)
        return x

"""
class Encoder(nn.Module):
	def __init__(self, input_shape, latent_dims):
		super(Encoder, self).__init__()

		channels, height, width = input_shape

		# Calculate output shape of image discriminator (PatchGAN)
		self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

		def discriminator_block(in_filters, out_filters, normalize=True):
			layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
			if normalize:
				layers.append(nn.InstanceNorm2d(out_filters))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*discriminator_block(channels, 64, normalize=False),
			*discriminator_block(64, 128),
			*discriminator_block(128, 256),
			*discriminator_block(256, 512),
			nn.ZeroPad2d((1, 0, 1, 0)),
			nn.Conv2d(512, 2*latent_dims + 1, 4, padding=1),
			nn.AvgPool2d(kernel_size=(16, 16)),
			nn.Flatten()
		)

		self.model = nn.Sequential(
			*discriminator_block(channels, 64, normalize=False),
			*discriminator_block(64, 128),
			*discriminator_block(128, 256),
			*discriminator_block(256, 512),
			nn.Conv2d(512, 100, 2, padding=1),
			nn.Flatten(),
			nn.ReLU(),
			nn.Linear(2500, 100),
			nn.ReLU(),
			nn.Linear(100, latent_dims + 1)
		)
		self.latent_dims = latent_dims

	def forward(self, img):
		out = self.model(img)
		# print(out.size())
		lat = out[:,:self.latent_dims]
		cl = nn.Sigmoid()(out[:, -1])
		cl = torch.reshape(cl, (-1, 1))
		return lat, cl
"""

"""
class Encoder(nn.Module):
	def __init__(self, input_shape, latent_dims):
		super(Encoder, self).__init__()

		self.conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(), nn.BatchNorm2d(32), nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),  nn.BatchNorm2d(64), nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.BatchNorm2d(128), nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.BatchNorm2d(128), nn.Conv2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)), nn.ReLU())
		self.bn1 = nn.BatchNorm1d(3*3*128)
		self.fc1 = nn.Linear(3*3*128, 2000 + 1)

	def forward(self, img):
		img = self.conv(img)
		#print(img.size())
		img = nn.Flatten()(img)
		img = self.bn1(img)
		img = self.fc1(img)
		cl = img[:, -1]
		lat = img[:, :-1]
		# print(out.size())
		return lat, cl
"""

class Encoder(nn.Module):
	def __init__(self, input_shape, latent_dims):
		super(Encoder, self).__init__()

		self.conv = nn.Sequential(ResBlockDown(input_channels=3), ResBlockDown(input_channels=128), ResBlockDown(input_channels=128), ResBlock(input_channels=128), ResBlock(input_channels=128), nn.BatchNorm2d(128), nn.Conv2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)), nn.ReLU())
		self.bn1 = nn.BatchNorm1d(3*3*128)
		self.fc1 = nn.Linear(3*3*128, 2000 + 1)

	def forward(self, img):
		img = self.conv(img)
		#print(img.size())
		img = nn.Flatten()(img)
		img = self.bn1(img)
		img = self.fc1(img)
		cl = img[:, -1]
		lat = img[:, :-1]
		# print(out.size())
		return lat, cl

"""
class Decoder(nn.Module):
	def __init__(self, latent_dims):
		super(Decoder, self).__init__()

		self.fc1 = nn.Linear(latent_dims, 100)
		self.fc2 = nn.Linear(100, 8*8*128)
		self.latent_dims = latent_dims
		self.deconvs = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

	def forward(self, lat):
		img = self.fc1(lat)
		img = nn.ReLU()(img)
		img = self.fc2(img)
		img = nn.ReLU()(img)
		img = torch.reshape(img, (-1, 128, 8, 8))
		out = self.deconvs(img)
		return out
"""

"""
class Decoder(nn.Module):
	def __init__(self, latent_dims):
		super(Decoder, self).__init__()

		self.fc1 = nn.Linear(2000 + 1, 4*4*128)
		self.bn1 = nn.BatchNorm1d(4*4*128)
		self.deconvs = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.BatchNorm2d(128), nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.BatchNorm2d(128), nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.BatchNorm2d(64), nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.BatchNorm2d(32), nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.Tanh())

	def forward(self, lat):
		img = self.fc1(lat)
		img = nn.ReLU()(img)
		img = self.bn1(img)
		#img = self.fc2(img)
		#img = nn.ReLU()(img)
		#img = self.bn2(img)
		img = torch.reshape(img, (-1, 128, 4, 4))
		out = self.deconvs(img)
		return out
"""

class Decoder(nn.Module):
	def __init__(self, latent_dims):
		super(Decoder, self).__init__()

		self.fc1 = nn.Linear(2000 + 1, 4*4*128)
		self.bn1 = nn.BatchNorm1d(4*4*128)
		self.deconvs = nn.Sequential(ResBlockUp(input_channels=128), ResBlockUp(input_channels=128), ResBlockUp(input_channels=128) ,ResBlockUp(input_channels=128), nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.Tanh())

	def forward(self, lat):
		img = self.fc1(lat)
		img = nn.ReLU()(img)
		img = self.bn1(img)
		#img = self.fc2(img)
		#img = nn.ReLU()(img)
		#img = self.bn2(img)
		img = torch.reshape(img, (-1, 128, 4, 4))
		out = self.deconvs(img)
		return out

class Classifier(nn.Module):
	def __init__(self, input_shape):
		super(Classifier, self).__init__()

		self.conv = nn.Sequential(ResBlockDown(input_channels=3, leak=True), ResBlockDown(input_channels=128, leak=True), ResBlockDown(input_channels=128, leak=True), ResBlock(input_channels=128), ResBlock(input_channels=128), nn.Conv2d(128, 10, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), nn.LeakyReLU(0.2))
		self.bn1 = nn.BatchNorm1d(4*4*512)
		self.fc1 = nn.Linear(6*6*10, 100)
		self.min1 = MinibatchDiscrimination(100, 30, 30)
		self.bn2 = nn.BatchNorm1d(100)
		self.fc2 = nn.Linear(130, 30)
		self.fc3 = nn.Linear(30, 2)
		self.fc4 = nn.Linear(30, 1)

	def forward(self, img):
		cl = self.conv(img)
		cl = nn.Flatten()(cl)
		cl = self.fc1(cl)
		cl = nn.LeakyReLU(0.2)(cl)
		cl = self.min1(cl)
		cl = self.fc2(cl)
		cl = nn.LeakyReLU(0.2)(cl)
		#print(cl.size())
		rl = self.fc4(cl)
		cl = self.fc3(cl)

		#print(img.size())
		#img = nn.Flatten()(img)
		#img = self.bn1(img)
		#img = nn.ReLU()(img)
		#img = self.fc1(img)
		#img = self.bn2(img)
		#img = nn.ReLU()(img)
		#cl = self.fc2(img)
		# print(out.size())
		return cl, rl


# Source: https://towardsdatascience.com/c2920e617dee
def load_ckp(model, optimizer=None, f_path='./best_model.pt'):
	# load check point
	checkpoint = torch.load(f_path)

	model.load_state_dict(checkpoint['state_dict'])

	optimizer.load_state_dict(checkpoint['optimizer'])

	valid_loss_min = checkpoint['valid_loss_min']
	epoch_train_loss = checkpoint['epoch_train_loss']
	epoch_valid_loss = checkpoint['epoch_valid_loss']

	return model, optimizer, checkpoint['epoch'], epoch_train_loss, epoch_valid_loss, valid_loss_min


class GenderClassifier(nn.Module):
	def __init__(self):
		super(GenderClassifier, self).__init__()

		self.conv = nn.Sequential(ResBlockDown(input_channels=3), ResBlockDown(input_channels=128), ResBlockDown(input_channels=128), ResBlock(input_channels=128), ResBlock(input_channels=128), nn.Conv2d(128, 10, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), nn.ReLU())
		self.bn1 = nn.BatchNorm1d(4*4*512)
		self.fc1 = nn.Linear(6*6*10, 100)
		self.dn1 = nn.Dropout(0.5)
		self.fc2 = nn.Linear(100, 30)
		self.dn2 = nn.Dropout(0.5)
		self.fc4 = nn.Linear(30, 2)

	def forward(self, img):
		cl = self.conv(img)
		cl = nn.Flatten()(cl)
		cl = self.fc1(cl)
		cl = nn.ReLU()(cl)
		cl = self.dn1(cl)
		cl = self.fc2(cl)
		cl = nn.ReLU()(cl)
		cl = self.dn2(cl)
		rl = self.fc4(cl)
		return rl
    
    
class BlondClassifier(nn.Module):
	def __init__(self):
		super(BlondClassifier, self).__init__()

		self.conv = nn.Sequential(ResBlockDown(input_channels=3), ResBlockDown(input_channels=128), ResBlockDown(input_channels=128), ResBlock(input_channels=128), ResBlock(input_channels=128), nn.Conv2d(128, 10, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), nn.ReLU())
		self.bn1 = nn.BatchNorm1d(4*4*512)
		self.fc1 = nn.Linear(6*6*10, 100)
		self.dn1 = nn.Dropout(0.5)
		self.fc2 = nn.Linear(100, 30)
		#self.fc3 = nn.Linear(30, 2)
		self.dn2 = nn.Dropout(0.5)
		self.fc4 = nn.Linear(30, 2)

	def forward(self, img):
		cl = self.conv(img)
		cl = nn.Flatten()(cl)
		cl = self.fc1(cl)
		cl = nn.ReLU()(cl)
		cl = self.dn1(cl)
		cl = self.fc2(cl)
		cl = nn.ReLU()(cl)
		cl = self.dn2(cl)
		#print(cl.size())
		rl = self.fc4(cl)
		#cl = self.fc3(cl)

		#print(img.size())
		#img = nn.Flatten()(img)
		#img = self.bn1(img)
		#img = nn.ReLU()(img)
		#img = self.fc1(img)
		#img = self.bn2(img)
		#img = nn.ReLU()(img)
		#cl = self.fc2(img)
		# print(out.size())
		return rl
    


def save_ckp(state, f_path='./best_model.pt'):
	torch.save(state, f_path)


def get_instance_model_optimizer(device, input_shape, learning_rate=0.0001, z_dims=2, pixel=64):
	print(device)
	classifier = Classifier(input_shape).to(device)
	#classifier = Classifier(input_shape)
	class_optim = torch.optim.Adam(classifier.parameters(), lr=learning_rate, betas=(0, 0.9))
	generator = Decoder(z_dims).to(device)
	#generator = Decoder(z_dims)
	gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0, 0.9))
	encoder = Encoder(input_shape, latent_dims=z_dims).to(device)
	#encoder = Encoder(input_shape, latent_dims=z_dims)
	enc_optim = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0, 0.9))
	discriminator = Discriminator(input_shape).to(device)
	#discriminator = Discriminator(input_shape)
	dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0, 0.9))
	
	return classifier, class_optim, generator, gen_optim, encoder, enc_optim, discriminator, dis_optim 


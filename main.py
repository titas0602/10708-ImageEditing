import torch
import numpy as np
from model_conv import get_instance_model_optimizer, load_ckp
from data import data_loader
from train import train_model
from evaluate import evaluate_model
import argparse
import warnings 
warnings.filterwarnings('ignore')
def train_and_evaluate(dataset_name,batch_size=100, n_epochs=5, learning_rate=0.0001, z_dim=2, pixel=64,
					   load_model=False):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	input_shape = (3, 256, 256)
	classifier, class_optim, generator, gen_optim, encoder, enc_optim, discriminator, dis_optim, scheduler_class, scheduler_gen, scheduler_enc, scheduler_dis = get_instance_model_optimizer(device, input_shape, learning_rate, z_dim,pixel, ep=n_epochs)
	train_loader, test_loader = data_loader(dataset_name, pixel, batch_size)
	if load_model:
		discriminator.load_state_dict(torch.load('Dis.pt'))
		generator.load_state_dict(torch.load('Gen.pt'))
		encoder.load_state_dict(torch.load('Enc.pt'))
		classifier.load_state_dict(torch.load('Class.pt'))
	else:
		valid_loss_min = np.inf
		start_epoch = 0
		epoch_train_loss = []
		epoch_valid_loss = []

		train_model(dataset_name, classifier, class_optim, generator, gen_optim, encoder, enc_optim, discriminator, dis_optim, train_loader, test_loader, device, start_epoch, n_epochs, epoch_train_loss,
					epoch_valid_loss, valid_loss_min, z_dim, pixel,batch_size, scheduler_class, scheduler_gen, scheduler_enc, scheduler_dis)

	evaluate_model(dataset_name, encoder, generator, z_dim, pixel,batch_size, device=device)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Train and Evaluate Harmony on your dataset')
	parser.add_argument('-z', '--z-dim', type=int, default=10)
	parser.add_argument('-bs', '--batch-size', type=int, default=100)
	parser.add_argument('-ep', '--num-epochs', type=int, default=50)
	parser.add_argument('-l', '--learning-rate', type=float, default=0.0002)
	parser.add_argument('--load-model', action='store_true')
	parser.add_argument('-dat','--dataset',type=str)
	parser.add_argument('-p','--pixel',type=int, default=28)
	args = parser.parse_args()
	num_epochs = args.num_epochs
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	z_dim = args.z_dim
	dataset_name = args.dataset
	pixel = args.pixel
	load_model = False
	if args.load_model:
		load_model = True
	dataset_name = '../CycleGAN/horse2zebra'
	train_and_evaluate(dataset_name=dataset_name,batch_size=batch_size, n_epochs=num_epochs, learning_rate=learning_rate, z_dim=z_dim, pixel=pixel, load_model=load_model)

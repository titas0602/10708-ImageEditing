import torch
from utils import *
from model_conv import save_ckp, load_ckp
from utils import plot_loss
from torch import nn
import torch.nn.functional as F
from queue import Queue
from tqdm import tqdm
from inception import InceptionV3
from scipy import linalg
from vgg_loss import VGGLoss

percept_loss = VGGLoss()

def plot_sample_images(images, recons, recons_fake):

	images = images.cpu().detach()
	recons = recons.cpu().detach()
	recons_fake = recons_fake.cpu().detach()
	batch_size = images.size(0)

	#print(torch.mean(torch.abs(images - recons)))

	for i in range(0, batch_size):
		r = recons[i]
		r = r - torch.min(r)
		r = r/torch.max(r)
		r = T.ToPILImage()(r)
		r.save('outputs1/'+str(i)+'.png')

	for i in range(0, batch_size):
		r = recons_fake[i]
		r = r - torch.min(r)
		r = r/torch.max(r)
		r = T.ToPILImage()(r)
		r.save('outputs1f/'+str(i)+'.png')

	for i in range(0, batch_size):
		r = images[i]
		r = r - torch.min(r)
		r = r/torch.max(r)
		r = T.ToPILImage()(r)
		r.save('inputs1/'+str(i)+'.png')

def reparametrize(mu, logvar):
	std = logvar.mul(0.5).exp_()
	eps = std.data.new(std.size()).normal_()
	return eps.mul(std).add_(mu)

def compute_discriminator_loss(discrim_real, discrim_fake_1, discrim_fake_2):
	#loss = 0.25*torch.mean(torch.square(discrim_fake_1)) + 0.25*torch.mean(torch.square(discrim_fake_2)) + 0.5*torch.mean(torch.square(1 - discrim_real))
	loss = 0.5*torch.mean(torch.square(discrim_fake_2)) + 0.5*torch.mean(torch.square(1 - discrim_real))
	return loss

def compute_generator_loss(discrim_fake_1, discrim_fake_2):
	# TODO 1.4.1: Implement LSGAN loss for generator.
	# discrim_fake = nn.Sigmoid()(discrim_fake)
	# loss = - torch.mean(torch.log(torch.clamp(discrim_fake, min=1e-4)))
	#loss = 0.5*torch.mean(torch.square(1 - discrim_fake_1)) + 0.5*torch.mean(torch.square(1 - discrim_fake_2))
	loss = torch.mean(torch.square(1 - discrim_fake_1))
	return loss

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)

	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	assert mu1.shape == mu2.shape, \
		'Training and test mean vectors have different lengths'
	assert sigma1.shape == sigma2.shape, \
		'Training and test covariances have different dimensions'

	diff = mu1 - mu2

	# Product might be almost singular
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		msg = ('fid calculation produces singular product; '
			   'adding %s to diagonal of cov estimates') % eps
		print(msg)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

	# Numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			m = np.max(np.abs(covmean.imag))
			raise ValueError('Imaginary component {}'.format(m))
		covmean = covmean.real

	tr_covmean = np.trace(covmean)

	return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_fid(batch1, batch2, model):
	pred1 = model(batch1)[0]
	pred2 = model(batch2)[0]
	pred1 = nn.Flatten()(pred1)
	pred2 = nn.Flatten()(pred2)
	mu1 = torch.mean(pred1, dim=0)
	mu2 = torch.mean(pred2, dim=0)
	sig1 = torch.cov(torch.transpose(pred1, 0, 1))
	sig2 = torch.cov(torch.transpose(pred2, 0, 1))
	mu1 = mu1.cpu().detach().numpy()
	mu2 = mu2.cpu().detach().numpy()
	sig1 = sig1.cpu().detach().numpy()
	sig2 = sig2.cpu().detach().numpy()
	fid = calculate_frechet_distance(mu1, sig1, mu2, sig2, eps=1e-6)
	return fid

def compute_discriminator_loss(discrim_real, discrim_fake, discrim_interp, interp, lamb):
	# TODO 1.5.1: Implement WGAN-GP loss for discriminator.
	# loss = max_D E[D(real_data)] - E[D(fake_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
	# discrim_fake = nn.Sigmoid()(discrim_fake)
	# discrim_real = nn.Sigmoid()(discrim_real)
	# discrim_interp = nn.Sigmoid()(discrim_interp)


	g = torch.autograd.grad(discrim_interp, interp, grad_outputs=torch.ones(discrim_interp.size()).cuda(), retain_graph=True, create_graph=True)
	# print(len(g))
	g = g[0]
	# print(g.size())
	g = torch.square(g)
	g = torch.sum(g, axis=(1, 2, 3))
	g = torch.pow(g, 0.5)
	g = g - 1
	g = torch.square(g)
	# print(g.size())

	loss = torch.mean(discrim_fake) - torch.mean(discrim_real) + lamb*torch.mean(g)
	return loss

def train_model(dataset_name, classifier, class_optim, generator, gen_optim, encoder, enc_optim, discriminator, dis_optim, train_loader, test_loader, device,
					   start_epoch, n_epochs, epoch_train_loss, epoch_valid_loss, valid_loss_min,
					   z_dim=2, pixel=64, batch_size=100, scheduler_class=None, scheduler_gen=None, scheduler_enc=None, scheduler_dis=None):

	K = 10
	epoch_valid_acc = []
	queue = []
	valid_acc_max = 0
	print('z_dim',z_dim)

	#block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
	#inception = InceptionV3([block_idx]).cuda()

	for epoch in tqdm(range(0, n_epochs + 1)):

		generator.train()
		discriminator.train()
		encoder.train()
		classifier.train()

		train_loss = 0.0
		valid_loss = 0.0
		dis_acc = 0
		class_acc = 0
		dec_acc = 0

		num_batches = 101
		for batch_idx, images in tqdm(enumerate(train_loader)):
			labels = images[1]
			images = images[0]

			images = images.cuda()
			labels = labels.cuda()
			labels = labels[:, 9].long()
			#print(labels)
			#labels = 2*labels - 1
			#labels = labels.reshape(images.size(0), 1)

			#print(labels)

			lat, cl = encoder(images)
			#lat = reparametrize(lat_mu, lat_var)
			recons = generator(torch.cat([lat, cl.unsqueeze(1)], 1))
			fake_recons = generator(torch.cat([lat, 1 - cl.unsqueeze(1)], 1))

			#Training the classifier
			recons_d = recons.detach()
			cl_im, rl_im = classifier(images + 0.1*torch.randn(images.size()).cuda())
			cl_recons, rl_recons = classifier(recons_d + 0.1*torch.randn(images.size()).cuda())
			fake_recons_d = fake_recons.detach()
			cl_fake, rl_fake = classifier(fake_recons_d + 0.1*torch.randn(images.size()).cuda())
			eps = torch.rand(images.size(0), 1, 1, 1).cuda()
			interp = eps*fake_recons + (1-eps)*images
			cl_interp, rl_interp = classifier(interp)
			#classifier_loss = torch.mean(torch.square(cl_im - labels)) + torch.mean(torch.square(cl_recons - labels)) + torch.mean(torch.square(cl_fake + 1))
			classifier_loss = compute_discriminator_loss(rl_im, rl_fake, rl_interp, interp, 10)
			classifier_loss = 10*classifier_loss + 10*nn.CrossEntropyLoss()(cl_im, labels)
			class_optim.zero_grad()
			classifier_loss.backward()
			class_optim.step()

			class_acc = class_acc + classifier_loss.cpu().detach().numpy()

			#Training the encoder
			lat, cl = encoder(images)
			recons = generator(torch.cat([lat, cl.unsqueeze(1)], 1))

			#Classifying fake images
			fake_recons = generator(torch.cat([lat, 1 - cl.unsqueeze(1)], 1))
			class_fake, rl_fake = classifier(fake_recons + 0.1*torch.randn(images.size()).cuda())

			#Cycle Consistency
			#lat_f, cl_f = encoder(fake_recons)
			#ake_recons_cycle = generator(torch.cat([lat_f, cl.unsqueeze(1)], 1))

			#Consistency loss
			recons_d = recons.detach()
			fake_d = fake_recons.detach()
			lat_r, cl_r = encoder(recons_d)
			lat_f, cl_f = encoder(fake_d)
			cons_loss = torch.mean(torch.square(lat_f - lat_r))

			acc_loss = torch.mean(torch.square(cl-labels))
			acc_loss_fake = nn.CrossEntropyLoss()(class_fake, 1 - labels) - torch.mean(rl_fake)
			loss = 250*torch.mean(torch.abs(recons - images)) + percept_loss(recons, images) + 10*acc_loss + 10*acc_loss_fake + 100*cons_loss

			enc_optim.zero_grad()
			loss.backward()
			enc_optim.step()

			#Training the decoder
			lat, cl = encoder(images)
			recons = generator(torch.cat([lat, cl.unsqueeze(1)], 1))

			#Classifying fake images
			fake_recons = generator(torch.cat([lat, 1 - cl.unsqueeze(1)], 1))
			class_fake, rl_fake = classifier(fake_recons + 0.1*torch.randn(images.size()).cuda())

			#Cycle Consistency
			#lat_f, cl_f = encoder(fake_recons)
			#fake_recons_cycle = generator(torch.cat([lat_f, cl.unsqueeze(1)], 1))

			#acc_loss = nn.CrossEntropyLoss()(cl, labels)
			acc_loss_fake = nn.CrossEntropyLoss()(class_fake, 1 - labels) - torch.mean(rl_fake)
			#loss = 150*torch.mean(torch.abs(recons - images)) + percept_loss(recons, images) + 10*acc_loss_fake + 100*cons_loss + 150*torch.mean(torch.abs(fake_recons_cycle - images))
			loss = 250*torch.mean(torch.abs(recons - images)) + percept_loss(recons, images) + 10*acc_loss_fake

			dec_acc = dec_acc + acc_loss_fake.cpu().detach().numpy()

			gen_optim.zero_grad()
			loss.backward()
			gen_optim.step()
			
			# print("Loss: {:.2f}".format(loss))
			train_loss += loss.item()
			#acc_loss = acc_loss.item()
			if batch_idx > 100:
				break

		epoch_train_loss.append(train_loss/101)
		plot_sample_images(images, recons, fake_recons)
		# validate the model #
		######################
		# print("Validating")
		torch.cuda.empty_cache()
		
		generator.eval()
		discriminator.eval()
		encoder.eval()
		classifier.eval()

		#scheduler_class.step()
		#scheduler_gen.step()
		#scheduler_enc.step()
		#scheduler_dis.step()

		#print(epoch_train_loss)
		#print(epoch_valid_loss)
		# print training/validation statistics
		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\t Discriminator Accuracy: {:.6f}\t Classifier Accuracy: {:.6f}\t Decoder Accuracy: {:.6f}\t'.format(
			epoch,
			epoch_train_loss[epoch],
			epoch_train_loss[epoch],
			dis_acc/(num_batches),
			class_acc/(num_batches),
			dec_acc/(num_batches)
		))
		torch.cuda.empty_cache()
		if epoch % 10 == 0:
			torch.save(encoder.state_dict(), 'Enc.pt')
			torch.save(discriminator.state_dict(), 'Dis.pt')
			torch.save(classifier.state_dict(), 'Class.pt')
			torch.save(generator.state_dict(), 'Gen.pt')

			#fid_tot = 0

			"""
			with torch.no_grad():
				for batch_idx, images in enumerate(test_loader):
					labels = images[1]
					images = images[0]

					images = images.cuda()
					labels = labels.cuda()

					#Training the discriminator
					
					lat_mu, lat_var, cl = encoder(images)
					lat = lat_mu
					# labels = 1 - labels
					labels = labels.float()
					labels = torch.reshape(labels, (labels.size(0), 1))
					recons = generator(torch.cat((lat, cl), 1))

					#fid_tot = fid_tot + get_fid(images, recons, inception) 
			"""

			#print(fid_tot/len(test_loader))



	plot_loss(epoch_train_loss=epoch_train_loss, epoch_valid_loss=epoch_valid_loss)


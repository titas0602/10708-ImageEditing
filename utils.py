import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import numpy as np
from model_conv import load_ckp
from scipy.stats import norm
from scipy.special import expit
import torchvision.transforms as T
from PIL import Image
from torch import nn

def get_rand_labels(batch_size, classes):
    rand_lab = F.one_hot(torch.arange(0, batch_size) % 10)
    indexes = torch.randperm(rand_lab.shape[0])
    rand_lab = rand_lab[indexes]
    return rand_lab

def rot_trans_loss(theta, trans, image):
    trans = F.tanh(trans)*4.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, D, H, W = image.size()
    center = torch.tensor([D/2, H / 2, W / 2]).repeat(B, 1).to(device=device)
    scale = torch.ones(B,1).to(device=device)
    angle = torch.rad2deg(theta)
    no_trans = torch.zeros(B,3).to(device=device)
    no_rot = torch.zeros(B,3).to(device=device)
    M = kornia.get_projective_transform(center=center, scales=scale, angles=angle)
    #print(M[0])
    affine_matrix = M[:,:,:3]
    #print(affine_matrix[0])
    loss = torch.mean(torch.sum(torch.square(trans), axis=1))
    loss = loss + torch.mean(3 - torch.diagonal(affine_matrix, dim1=-2, dim2=-1).sum(-1))*0.4
    #print(loss.size())
    loss = 4*np.pi*loss/3
    return loss

def rot_trans(phi1, phi2, image):
    return rot_trans_loss(phi1[:,:3], phi1[:,3:6], image) + rot_trans_loss(phi2[:,:3], phi2[:,3:6], image)

def vae_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def fourier_mse(im1, im2):
    A = torch.fft(im1*im1)
    B = torch.fft(im2*im2)
    AB = torch.fft(im1*im2)
    f = A + B - 2*AB
    fmse = torch.sum(torch.fft.ifft(f).float())
    return fmse

def loss_fn(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, dim):
    n = image_x_theta1.size(0)
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)
    #recon_loss1 = fourier_mse(image_z1, image_x_theta1).div(n)
    #recon_loss2 = fourier_mse(image_z2, image_x_theta2).div(n)
    #branch_loss = fourier_mse(image_x_theta1, image_x_theta2).div(n)
    z1_mean = phi1[:,6:6+dim]
    z1_var = phi1[:, -dim:]
    z2_mean = phi2[:,6:6+dim]
    z2_var = phi2[:,-dim:]
    dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(z1_var.exp()))
    dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(z2_var.exp()))
    z_loss = torch.mean(torch.distributions.kl.kl_divergence(dist_z1, dist_z2)).div(dim) 
    #loss = 10*(recon_loss1 + recon_loss2 + branch_loss) + z_loss
    loss = 10*recon_loss1 + 10*recon_loss2 + z_loss
    return loss

def bhattacharyya_coefficient(mu1, mu2, logvar1, logvar2):
    eps = 1e-9
    var1 = torch.diag_embed(torch.clamp(logvar1.exp(), max=1/eps) + eps)
    var2 = torch.diag_embed(torch.clamp(logvar2.exp(), max=1/eps) + eps)
    avg_var = 0.5 * (var1 + var2)
    inv_avg_var = torch.linalg.inv(avg_var)
    diff_mean = mu1 - mu2
    db = 1/8 * torch.sum(torch.matmul(diff_mean, inv_avg_var) * diff_mean, dim=-1) # n_priors, n_priors
    
    """
    if not torch.all(torch.diagonal(avg_var, dim1=-2, dim2=-1) + eps > 0):
        print(torch.diagonal(avg_var, dim1=-2, dim2=-1)+eps)
        time.sleep(600)
    """


    db = db + 0.5 * (torch.sum(torch.log(torch.diagonal(avg_var, dim1=-2, dim2=-1)), dim=-1) - 0.5 * (torch.sum(logvar1, dim=-1) + torch.sum(logvar2, dim=-1)))
    #db = db - 0.5 * (torch.sum(logvar1, dim=-1) + torch.sum(logvar2, dim=-1))

    #bc = torch.exp(-db)
    #valid_bc = bc.mul(mask)
    bc = torch.sum(db)
    return bc

def cross_corr_loss(x, y):
    #print(x.size())
    xm = torch.mean(x, (-3, -2, -1), keepdim=True)
    ym = torch.mean(y, (-3, -2, -1), keepdim=True)
    #print(xm.size())
    x = x - xm
    y = y - ym
    num = torch.sum(x*y, (-4, -3, -2, -1))
    den = torch.sqrt(torch.sum(torch.square(x), (-4, -3, -2, -1)))*torch.sqrt(torch.sum(torch.square(y), (-4, -3, -2, -1)))
    loss = num/den
    loss = 1 - loss
    return torch.sum(loss)

def loss_fn_neg_div_bc(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, dim): # Sim Loss/Dissim Loss
    n = image_x_theta1.size(0)
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)
    #recon_loss1 = fourier_mse(image_z1, image_x_theta1).div(n)
    #recon_loss2 = fourier_mse(image_z2, image_x_theta2).div(n)
    #branch_loss = fourier_mse(image_x_theta1, image_x_theta2).div(n)
    eps = 1e-9
    #print(phi1)
    phi1 = torch.clamp(phi1, max=-eps) + torch.clamp(phi1, min=eps)
    #print(phi1)
    phi2 = torch.clamp(phi2, max=-eps) + torch.clamp(phi2, min=eps)
    z1_mean = phi1[:,6:6+dim]
    z1_var = phi1[:, -dim:]
    z2_mean = phi2[:,6:6+dim]
    z2_var = phi2[:,-dim:]
    #dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(torch.clamp(z1_var.exp(), min = eps)))
    #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
    #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
    z_loss = bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
    neg_z_loss = 0
    batch_size = z1_mean.size(0)
    #batch_size = 2
    for i in range(0,batch_size-1):
        z2_mean = torch.roll(z2_mean, 1, 0)
        z2_var = torch.roll(z2_var, 1, 0)
        #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
        #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
        if i==0:
            neg_z_loss = bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
        else:
            neg_z_loss = neg_z_loss + bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
        #print(neg_z_loss[:10])
    z_loss = batch_size*torch.mean(z_loss/torch.clip(neg_z_loss - 0.14*batch_size*z_loss, min=1e-6))
    #z_loss = batch_size*torch.mean(z_loss/neg_z_loss)
    loss = 10*(recon_loss1 + recon_loss2 + branch_loss) + 100*z_loss
    #print(loss)
    #print(z_loss)
    #print(neg_z_loss/batch_size)
    return loss

def loss_fn_neg_div_bc_class(image_z1, image_z2, image_x_theta1, image_x_theta2, z1_var, z1_mu, z2_var, z2_mu, class1, class2, dim): # Sim Loss/Dissim Loss
    n = image_x_theta1.size(0)*32*32*32
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)
    #recon_loss1 = fourier_mse(image_z1, image_x_theta1).div(n)
    #recon_loss2 = fourier_mse(image_z2, image_x_theta2).div(n)
    #branch_loss = fourier_mse(image_x_theta1, image_x_theta2).div(n)
    eps = 1e-9
    #print(phi1)
    #phi1 = torch.clamp(phi1, max=-eps) + torch.clamp(phi1, min=eps)
    #print(phi1)
    #phi2 = torch.clamp(phi2, max=-eps) + torch.clamp(phi2, min=eps)
    #z1_mean = phi1[:,6:6+dim]
    #z1_var = phi1[:, -dim:]
    #z2_mean = phi2[:,6:6+dim]
    #z2_var = phi2[:,-dim:]
    z1_mu = torch.clamp(z1_mu, max=-eps) + torch.clamp(z1_mu, min=eps)
    z2_mu = torch.clamp(z2_mu, max=-eps) + torch.clamp(z2_mu, min=eps)
    z1_var = torch.clamp(z1_var, max=-eps) + torch.clamp(z1_var, min=eps)
    z2_var = torch.clamp(z2_var, max=-eps) + torch.clamp(z2_var, min=eps)
    #dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(torch.clamp(z1_var.exp(), min = eps)))
    #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
    #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
    z_loss = bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
    neg_z_loss = 0
    batch_size = z1_mean.size(0)
    #batch_size = 2
    #class_pred_1 = torch.argmax(class1, dim=1)
    #class_pred_1.requires_grad = False
    #class_pred_2 = torch.argmax(class1, dim=1)
    #class_pred_2.requires_grad = False
    for i in range(0,batch_size-1):
        z2_mean = torch.roll(z2_mean, 1, 0)
        z2_var = torch.roll(z2_var, 1, 0)
        #class_pred_2 = torch.roll(class_pred_2, 1, 0)
        #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
        #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
        #print(class_pred_1)
        if i==0:
            neg_z_loss = bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
        else:
            neg_z_loss = neg_z_loss + bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
        #print(neg_z_loss[:10])
    #z_loss = batch_size*torch.mean(z_loss/torch.clip(neg_z_loss - 0.14*batch_size*z_loss, min=1e-6))
    z_loss = batch_size*torch.mean(z_loss/neg_z_loss)

    v1 = vae_loss(z1_mean, z1_var)
    v2 = vae_loss(z1_mean, z1_var)

    loss = 10*(recon_loss1) + 100*z_loss + 100*(v1 + v2)
    #print(loss)
    #print(z_loss)
    #print(neg_z_loss/batch_size)
    return loss

def loss_fn_neg_div_bc_class_nh(image_z1, image_z2, image_x_theta1, image_x_theta2, z1_var, z1_mu, z2_var, z2_mu, class1, class2, dim): # Sim Loss/Dissim Loss
    n = image_x_theta1.size(0)*32*32*32
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)
    #recon_loss1 = fourier_mse(image_z1, image_x_theta1).div(n)
    #recon_loss2 = fourier_mse(image_z2, image_x_theta2).div(n)
    #branch_loss = fourier_mse(image_x_theta1, image_x_theta2).div(n)
    eps = 1e-9
    #print(phi1)
    #phi1 = torch.clamp(phi1, max=-eps) + torch.clamp(phi1, min=eps)
    #print(phi1)
    #phi2 = torch.clamp(phi2, max=-eps) + torch.clamp(phi2, min=eps)
    #z1_mean = phi1[:,6:6+dim]
    #z1_var = phi1[:, -dim:]
    #z2_mean = phi2[:,6:6+dim]
    #z2_var = phi2[:,-dim:]
    z1_mu_sign = torch.sign(z1_mu)
    z1_mu_sign[z1_mu_sign==0] = 1
    z1_mean = torch.clamp(torch.abs(z1_mu), min=eps)*z1_mu_sign

    z2_mu_sign = torch.sign(z2_mu)
    z2_mu_sign[z2_mu_sign==0] = 1
    z2_mean = torch.clamp(torch.abs(z2_mu), min=eps)*z2_mu_sign

    #dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(torch.clamp(z1_var.exp(), min = eps)))
    #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
    #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
    z_loss = bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
    # print(z_loss)
    """
    neg_z_loss = 0
    batch_size = z1_mean.size(0)
    #batch_size = 2
    #class_pred_1 = torch.argmax(class1, dim=1)
    #class_pred_1.requires_grad = False
    #class_pred_2 = torch.argmax(class1, dim=1)
    #class_pred_2.requires_grad = False
    for i in range(0,batch_size-1):
        z2_mean = torch.roll(z2_mean, 1, 0)
        z2_var = torch.roll(z2_var, 1, 0)
        #class_pred_2 = torch.roll(class_pred_2, 1, 0)
        #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
        #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
        #print(class_pred_1)
        if i==0:
            neg_z_loss = bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
        else:
            neg_z_loss = neg_z_loss + bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
        #print(neg_z_loss[:10])
    #z_loss = batch_size*torch.mean(z_loss/torch.clip(neg_z_loss - 0.14*batch_size*z_loss, min=1e-6))
    z_loss = batch_size*torch.mean(z_loss/torch.clip(neg_z_loss, min=1e-9))
    """
    #loss = 300*(recon_loss1) + 100*z_loss
    #loss = 300*recon_loss1
    loss = 300*(recon_loss1) + z_loss
    #print(loss)
    #print(z_loss)
    #print(neg_z_loss/batch_size)
    return loss

def loss_fn_neg_div_bc_moco(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, class1, class2, dim, z_q, z_k, queue, K): # Sim Loss/Dissim Loss
    n = image_x_theta1.size(0)*32*32*32
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)
    #recon_loss1 = fourier_mse(image_z1, image_x_theta1).div(n)
    #recon_loss2 = fourier_mse(image_z2, image_x_theta2).div(n)
    #branch_loss = fourier_mse(image_x_theta1, image_x_theta2).div(n)
    eps = 1e-9
    #print(phi1)
    phi1 = torch.clamp(phi1, max=-eps) + torch.clamp(phi1, min=eps)
    #print(phi1)
    phi2 = torch.clamp(phi2, max=-eps) + torch.clamp(phi2, min=eps)
    z1_mean = phi1[:,6:6+dim]
    z1_var = phi1[:, -dim:]
    z2_mean = phi2[:,6:6+dim]
    z2_var = phi2[:,-dim:]
    #dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(torch.clamp(z1_var.exp(), min = eps)))
    #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
    #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
    batch_size = z1_mean.size(0)

    pos_loss = torch.exp(torch.bmm(z_q.view(batch_size, 1, dim), z_k.view(batch_size, dim, 1)))
    
    ind = 0
    for it in queue:
        if ind==0:
            neg_loss = torch.exp(torch.mm(z_q.view(batch_size, dim), torch.transpose(it, 0, 1)))
        else:
            neg_loss = neg_loss + torch.exp(torch.mm(z_q.view(batch_size, dim), torch.transpose(it, 0, 1)))
            ind = 1
    if len(queue) > 0:
        #print(neg_loss.size())
        z_loss = -10*torch.mean(torch.log(torch.sum(pos_loss, axis=(1,2))/torch.sum(neg_loss, axis=1)))
    else:
        z_loss = 0
    loss = 10*(recon_loss1 + recon_loss2 + branch_loss) + z_loss
    #print(loss)
    #print(100*z_loss)
    #print(neg_z_loss/batch_size)
    return loss

def loss_fn_neg_div_bc_moco_nh(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, class1, class2, dim, z_q, z_k, queue, K): # Sim Loss/Dissim Loss
    n = image_x_theta1.size(0)*32*32*32
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)
    #recon_loss1 = fourier_mse(image_z1, image_x_theta1).div(n)
    #recon_loss2 = fourier_mse(image_z2, image_x_theta2).div(n)
    #branch_loss = fourier_mse(image_x_theta1, image_x_theta2).div(n)
    eps = 1e-9
    #print(phi1)
    phi1 = torch.clamp(phi1, max=-eps) + torch.clamp(phi1, min=eps)
    #print(phi1)
    phi2 = torch.clamp(phi2, max=-eps) + torch.clamp(phi2, min=eps)
    z1_mean = phi1[:,6:6+dim]
    z1_var = phi1[:, -dim:]
    z2_mean = phi2[:,6:6+dim]
    z2_var = phi2[:,-dim:]
    #dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(torch.clamp(z1_var.exp(), min = eps)))
    #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
    #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
    batch_size = z1_mean.size(0)

    pos_loss = torch.exp(torch.bmm(z_q.view(batch_size, 1, dim), z_k.view(batch_size, dim, 1)))
    
    ind = 0
    for it in queue:
        if ind==0:
            neg_loss = torch.exp(torch.mm(z_q.view(batch_size, dim), torch.transpose(it, 0, 1)))
        else:
            neg_loss = neg_loss + torch.exp(torch.mm(z_q.view(batch_size, dim), torch.transpose(it, 0, 1)))
            ind = 1
    if len(queue) > 0:
        #print(neg_loss.size())
        z_loss = -10*torch.mean(torch.log(torch.sum(pos_loss, axis=(1,2))/torch.sum(neg_loss, axis=1)))
    else:
        z_loss = 0
    loss = 100*(recon_loss1 + recon_loss2 + branch_loss) + z_loss
    #print(loss)
    #print(100*z_loss)
    #print(neg_z_loss/batch_size)
    return loss

def loss_fn_neg_div_bc_cross_corr(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, dim): # Sim Loss/Dissim Loss
    n = image_x_theta1.size(0)
    recon_loss1 = cross_corr_loss(image_z1, image_x_theta1).div(n)
    recon_loss2 = cross_corr_loss(image_z2, image_x_theta2).div(n)
    branch_loss = cross_corr_loss(image_x_theta1, image_x_theta2).div(n)
    #recon_loss1 = fourier_mse(image_z1, image_x_theta1).div(n)
    #recon_loss2 = fourier_mse(image_z2, image_x_theta2).div(n)
    #branch_loss = fourier_mse(image_x_theta1, image_x_theta2).div(n)
    eps = 1e-9
    #print(phi1)
    phi1 = torch.clamp(phi1, max=-eps) + torch.clamp(phi1, min=eps)
    #print(phi1)
    phi2 = torch.clamp(phi2, max=-eps) + torch.clamp(phi2, min=eps)
    z1_mean = phi1[:,6:6+dim]
    z1_var = phi1[:, -dim:]
    z2_mean = phi2[:,6:6+dim]
    z2_var = phi2[:,-dim:]
    #dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(torch.clamp(z1_var.exp(), min = eps)))
    #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
    #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
    z_loss = bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
    neg_z_loss = 0
    batch_size = z1_mean.size(0)
    #batch_size = 2
    for i in range(0,batch_size-1):
        z2_mean = torch.roll(z2_mean, 1, 0)
        z2_var = torch.roll(z2_var, 1, 0)
        #dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(torch.clamp(z2_var.exp(), min=eps)))
        #dist_m = torch.distributions.multivariate_normal.MultivariateNormal(0.5*(z1_mean + z2_mean), torch.diag_embed(torch.clamp(0.25*(z1_var.exp() + z2_var.exp()), min=eps)))
        if i==0:
            neg_z_loss = bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
        else:
            neg_z_loss = neg_z_loss + bhattacharyya_coefficient(z1_mean, z2_mean, z1_var, z2_var).div(dim)
        #print(neg_z_loss[:10])
    z_loss = batch_size*torch.mean(z_loss/neg_z_loss)
    loss = 10*(recon_loss1 + recon_loss2 + branch_loss) + z_loss
    #print(loss)
    #print(z_loss)
    #print(neg_z_loss/batch_size)
    return loss

def loss_fn_cross_corr(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, dim):
    n = image_x_theta1.size(0)
    recon_loss1 = cross_corr_loss(image_z1, image_x_theta1).div(n)
    recon_loss2 = cross_corr_loss(image_z2, image_x_theta2).div(n)
    branch_loss = cross_corr_loss(image_x_theta1, image_x_theta2).div(n)
    #recon_loss1 = fourier_mse(image_z1, image_x_theta1).div(n)
    #recon_loss2 = fourier_mse(image_z2, image_x_theta2).div(n)
    #branch_loss = fourier_mse(image_x_theta1, image_x_theta2).div(n)
    z1_mean = phi1[:,6:6+dim]
    z1_var = phi1[:, -dim:]
    z2_mean = phi2[:,6:6+dim]
    z2_var = phi2[:,-dim:]
    dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(z1_var.exp()))
    dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(z2_var.exp()))
    z_loss = torch.mean(torch.distributions.kl.kl_divergence(dist_z1, dist_z2)).div(dim) 
    loss = 10*(recon_loss1 + recon_loss2 + branch_loss) + z_loss
    #print(z_loss)
    #print(recon_loss1 + recon_loss2 + branch_loss)
    return loss

def chain_loss_fn(image_z1, image_z2, image_x_theta1, image_x_theta2, image_z1_f, image_z2_f, image_x_theta1_f, image_x_theta2_f):
    n = image_z1.size(0)
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)
    loss1 = recon_loss1 + recon_loss2 + branch_loss
    recon_loss1_f = F.mse_loss(image_z1_f, image_x_theta1_f, reduction="sum").div(n)
    recon_loss2_f = F.mse_loss(image_z2_f, image_x_theta2_f, reduction="sum").div(n)
    branch_loss_f = F.mse_loss(image_x_theta1_f, image_x_theta2_f)
    loss1_f = recon_loss1_f + recon_loss2_f + branch_loss_f
    return loss1 + loss1_f

def plot_loss(epoch_train_loss, epoch_valid_loss):
    fig, ax = plt.subplots(dpi=150)
    train_loss_list = [x for x in epoch_train_loss]
    valid_loss_list = [x for x in epoch_valid_loss]
    line1, = ax.plot([i for i in range(len(train_loss_list))], train_loss_list)
    line2, = ax.plot([i for i in range(len(valid_loss_list))], valid_loss_list)
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Loss')
    ax.legend((line1, line2), ('Train Loss', 'Validation Loss'))
    plt.savefig("Harmony_mnist_loss_curves.png", bbox_inches="tight")


def _save_sample_images(dataset_name,batch_size, recon_image, image,pixel, mu=None, std=None):
    sample_out = recon_image.reshape(batch_size, pixel*pixel).astype(np.float32)
    #sample_out = np.expand_dims(mu,axis=1).repeat(pixel*pixel*pixel,axis=1) + sample_out*np.expand_dims(std,axis=1).repeat(pixel*pixel*pixel,axis=1) 
    #print(sample_out.shape, mu.shape, std.shape)
    #minis = sample_out.min(axis=1)
    #maxis = sample_out.max(axis=1)
    #sample_out = (sample_out - minis[:,np.newaxis])/ (maxis[:,np.newaxis] - minis[:,np.newaxis])
    sample_out = sample_out.reshape(batch_size, pixel, pixel)
    plt.clf()
    fig = plt.figure(figsize=(10, 10))  # Notice the equal aspect ratio
    plot_per_row = 10
    plot_per_col = 10
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(100)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        a.imshow(sample_out[i], cmap='binary')
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Harmony_decoded_image_sample_"+dataset_name+".png", bbox_inches="tight")

    sample_in = image.reshape(batch_size, pixel*pixel).astype(np.float32)
    #sample_in = np.expand_dims(mu,axis=1).repeat(pixel*pixel*pixel,axis=1) + sample_in*np.expand_dims(std,axis=1).repeat(pixel*pixel*pixel,axis=1) 
    sample_in = sample_in.reshape(batch_size, pixel, pixel)
    plt.clf()
    fig = plt.figure(figsize=(10, 10))  # Notice the equal aspect ratio
    plot_per_row = 10
    plot_per_col = 10
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(100)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        a.imshow(sample_in[i], cmap='binary')
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Harmony_input_image_sample"+dataset_name+".png", bbox_inches="tight")

def generate_manifold_images(dataset_name, trained_vae, pixel, z_dim=1, batch_size=100, device='cuda'):
    trained_vae.eval()
    decoder = trained_vae.autoencoder.decoder
    if z_dim==1:
        z_arr = norm.ppf(np.linspace(0.05, 0.95, batch_size))
    else:
        n = int(np.sqrt(batch_size))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        z_list = []
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                ls = [xi*20, yi*20]
                z_sample = np.array([ls])
                z_list.append(z_sample)
                #ls = [grid_x[6], grid_x[6], grid_x[6], grid_x[6], grid_x[6], grid_x[6], grid_x[6], grid_x[6], grid_x[6], grid_x[6]]
                #ls[j] = yi
                #z_sample = np.array([ls])
                #z_list.append(z_sample)
        z_arr = np.array(z_list).squeeze()

    z = torch.from_numpy(z_arr).float().to(device=device)
    if z_dim==1:
        z = torch.unsqueeze(z, 1)
    print(z.shape)
    image_z = decoder(z)
    image_z = F.sigmoid(image_z)
    manifold = image_z.cpu().detach().numpy()
    print(np.shape(manifold))
    sample_out = manifold.reshape(batch_size, pixel, pixel).astype(np.float32)
    #sample_out = np.transpose(manifold, axes=(0, 3, 2, 1))
    #sample_out = manifold.reshape(batch_size, 3, pixel, pixel).astype(np.float32)
    plt.clf()
    fig = plt.figure(figsize=(10, 10))  # Notice the equal aspect ratio
    plot_per_row = int(batch_size / 10)
    plot_per_col = int(batch_size / 10)
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        #sample_out[i] = sample_out[i] - np.min(sample_out[i])
        #sample_out[i] = sample_out[i]/np.max(sample_out[i])
        a.imshow(sample_out[i], cmap='binary')
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Harmony_manifold_image_"+dataset_name+".png", bbox_inches="tight")


def plot_sample_images(dataset_name, test_loader, encoder, generator, pixel, batch_size, device):
    encoder.eval()
    generator.eval()
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            labels = images[1]
            images = images[0]

            images = images.cuda()
            labels = labels.cuda()

            #Training the discriminator
            
            lat = encoder(images)
            #labels = 1 - labels
            #labels = labels.float()
            #labels = torch.reshape(labels, (labels.size(0), 1))
            recons = generator(lat)

            break

    images = images.cpu().detach()
    recons = recons.cpu().detach()

    for i in range(0, batch_size):
        r = recons[i]
        r = r - torch.min(r)
        r = r/torch.max(r)
        r = T.ToPILImage()(r)
        r.save('outputs/'+str(i)+'.png')

    for i in range(0, batch_size):
        r = images[i]
        r = r - torch.min(r)
        r = r/torch.max(r)
        r = T.ToPILImage()(r)
        r.save('inputs/'+str(i)+'.png')



    

def save_sample_images(dataset_name,test_loader, trained_model, pixel, batch_size=100, device='cuda'):
    trained_model.eval()
    for batch_idx, images in enumerate(test_loader):
        with torch.no_grad():
            labels = images[1]
            images = images[0]
            images = images.float()
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, class1, class2 = trained_model(data)
    pose_image = image_z1.cpu().detach().numpy()
    f = open('output_images_'+dataset_name+'_test.pkl','wb')
    pickle.dump(pose_image,f)
    f.close()

def save_latent_variables(dataset_name,data_loader, siamese, type, pixel, batch_size=100, device='cuda'):
    Allphi = []
    siamese.eval()
    count = 0
    for batch_idx, images in enumerate(data_loader):
        count +=1
        with torch.no_grad():
            labels = images[1]
            images = images[0]
            images = images.float()
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, class1, class2 = siamese(data)
            phi_np = phi1.cpu().detach().numpy()
            Allphi.append(phi_np)
    # Save the latent variables and theta
    #print(PhiArr.shape)
    PhiArr = np.array(Allphi).reshape(count*batch_size,-1)
    filepath = 'saved_latents/Harmony_z_dim_1_phi_values_'+dataset_name +'_'+ type + '.np'
    np.savetxt(filepath,PhiArr)

import h5py
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import wandb
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import get_loader
from model import CelebA_EBM
from utils import *


class Args:
    batch_size = 128
    lr = 1e-3
    image_dir = '/mnt/data/10708-controllable-generation/data/celeba/img_align_celeba'
    attr_fp = '/mnt/data/10708-controllable-generation/data/celeba/list_attr_celeba.txt'
    seed = 208975
    num_workers = 2
    use_wandb = False
    n_epochs = 7
    
    
def train(n_epochs, model, device, optimizer, train_loader, val_loader, use_wandb=False):
    losses = []
    for epoch in range(n_epochs):
        
        total_loss = 0
        batch_cnt = 0
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels[:, -2].to(device)
            correct_out = model(images, labels)
            wrong_out = model(images, 1 - labels)
            loss = (wrong_out + 1).square().mean() + (correct_out - 1).square().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach().cpu().item()
            batch_cnt += 1
        
        loss = total_loss / batch_cnt
        print(f'Epoch: {epoch}/{n_epochs}\tTrain Loss: {loss:.4}')
        torch.save(model, f'saved_models/epoch{epoch}.pth')
        losses.append(loss)
    plt.plot(loss)
    plt.savefig('train_loss.png')
            
    
def main():
    args = Args()
    if args.use_wandb:
        wandb.init(project='pgm_project', reinit=True)
        run_name = wandb.run.name
    set_seed(args.seed)
    
    train_loader = get_loader(args.image_dir, 
                           args.attr_fp, 
                           ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
                           178, 128, 32,
                           'CelebA', 'train', 2)
    val_loader = get_loader(args.image_dir, 
                           args.attr_fp, 
                           ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
                           178, 128, 32,
                           'CelebA', 'train', 2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CelebA_EBM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train(args.n_epochs, model, device, optimizer, train_loader, val_loader, 
          use_wandb=args.use_wandb)
    
    torch.save(model, 'model.pth')
    
    return


if __name__ == '__main__':
    main()
import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from data_loader import *
from model_conv import *

def main():
    train_loader = get_loader( '/mnt/data/10708-controllable-generation/data/celeba/img_align_celeba', 
                           '/mnt/data/10708-controllable-generation/data/celeba/list_attr_celeba.txt', 
                           ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
                           114, 64, 32,
                           'CelebA', 'train', 2)
    test_loader = get_loader( '/mnt/data/10708-controllable-generation/data/celeba/img_align_celeba', 
                           '/mnt/data/10708-controllable-generation/data/celeba/list_attr_celeba.txt', 
                           ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
                           114, 64, 32,
                           'CelebA', 'test', 2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GenderClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    n_epochs = 10
    for epoch in range(n_epochs):
        
        total_loss = 0
        batch_cnt = 0
        # train
        model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device).float(), labels.to(device).long()
            logits = model(images)
            targs = labels[:, -2]
            loss = criterion(logits, targs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach().item()
            batch_cnt += 1
            
        print(f'Epoch: {epoch}')
        print(f'Train loss: {total_loss/batch_cnt:.4}')
            
        model.eval()
        total_loss = 0
        batch_cnt = 0
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device).float(), labels.to(device).long()
            with torch.no_grad():
                logits = model(images)
            targs = labels[:, -2]
            loss = criterion(logits, targs)
            total_loss += loss.detach().item()
            batch_cnt += 1
        print(f'Test loss: {total_loss/batch_cnt:.4}')
        
        torch.save(model, f'saved_models/epoch{epoch}.pth')
        
    
if __name__ == '__main__':
    main()
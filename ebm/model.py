import torch
import torch.nn as nn


class EBM_Binary(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.image_encoder = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(64*64, 20),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ])
        self.head = nn.Sequential(*[
            nn.Linear(21, 10),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(10, 1)
        ])
        
    def forward(self, image, shape):
        if len(shape.shape) == 1:
            shape = shape.unsqueeze(1)
        image_embedding = self.image_encoder(image)
        cat_inp = torch.cat((image_embedding, shape), dim=1)
        out = self.head(cat_inp)
        return out


class DSprites_EBM_Binary(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.image_encoder = nn.Sequential(*[
            nn.Conv2d(1, 4, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        ])
        self.head = nn.Sequential(*[
            nn.Linear(1569, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        ])
        
    def forward(self, image, shape):
        if len(shape.shape) == 1:
            shape = shape.unsqueeze(1)
        image_embedding = self.image_encoder(image)
        cat_inp = torch.cat((image_embedding, shape), dim=1)
        out = self.head(cat_inp)
        return out


class CelebA_EBM(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.image_encoder = nn.Sequential(*[
            nn.Conv2d(3, 4, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(4, 4, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        ])
        self.head = nn.Sequential(*[
            nn.Linear(3601, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 1)
        ])
        
    def forward(self, image, label):
        image_embedding = self.image_encoder(image)
        # print(image_embedding.shape)
        # print(label.shape)
        label = label.unsqueeze(1)
        cat_inp = torch.cat((image_embedding, label), dim=1)
        out = self.head(cat_inp)
        return out
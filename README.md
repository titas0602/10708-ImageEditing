# 10708-ImageEditing

This repo contains work done for the course project of 10-708 at CMU. To train the model, please use the following command:-

```
python3 main.py --num-epochs 100 --batch-size 100
```

Set the number of epochs and batch size according to your convenience. The model by default is trained for converting hair color between blonde/black. To filter the input to the dataloader, change lines 39-59 in data.py. To change the task being performed, change the index in line 169 in train.py. It is set to 9 currently which indicates blonde in the CelebA dataset. For the dataset, it is available at https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Please download it and specify the path to the dataset at lines 35 and 36 in data.py

To use pretrained models, please use the following command:-

```
python3 main.py --num-epochs 100 --batch-size 100 --load-model
```

The code saves pretrained models in the same folder as main.py


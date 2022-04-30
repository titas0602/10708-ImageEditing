# 10708-ImageEditing

This repo contains work done for the course project of 10-708 at CMU.

### Data

1. We use the CelebA dataset for this project. To prepare the dataset, first download `img_align_celeba.zip` from https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ.
2. Run the following 

```
unzip img_align_celeba.zip
mkdir data
mv img_align_celeba data/

```
3. Download all the `.txt` files from https://drive.google.com/drive/folders/1cMbvvEhsd0qu6dNt-Nlda9s-YCZgjkc0?usp=sharing into `data/`

These files will be required to train and evaluate the methods in the repo and the instructions for each can be found in the respective subdirectories.

### Model checkpoints

Relevant model checkpoints can be found at: https://drive.google.com/drive/folders/1cMbvvEhsd0qu6dNt-Nlda9s-YCZgjkc0?usp=sharing

### File Structure

```
ebm/ -- contains code for Energy-Based Model experiments
encoder-decoder/ -- contains code for our proposed method
stargan/ -- contains code for Star-GAN (our baseline)
auxiliary_models/ -- contains code for blonde and male classifiers required for evaluation
```


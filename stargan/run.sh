
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                 --celeba_image_dir '/mnt/data/10708-controllable-generation/data/celeba/img_align_celeba' \
                 --attr_path '/mnt/data/10708-controllable-generation/data/celeba/list_attr_celeba.txt' \
                 --model_save_dir='stargan_celeba_128/models' \
                 --result_dir='stargan_celeba_128/results'
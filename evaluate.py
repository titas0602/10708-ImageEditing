from data import data_loader, loadpickle
from utils import  save_latent_variables,  generate_manifold_images, save_sample_images
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from model_conv import *
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2

def class_acc(encoder, generator, loader, batch_size):

    classifier = BlondClassifier()
    #classifier.load_state_dict(torch.load('Blond.pt'))
    classifier = torch.load('epoch5.pth')
    classifier = classifier.cuda()

    class_acc = 0

    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            labels = images[1]
            images = images[0]

            images = images.cuda()
            labels = labels.cuda()
            labels = labels[:, 9].long()

            #Training the discriminator
            lat, cl = encoder(images)
            fake_recons = generator(torch.cat([lat, 1 - cl.unsqueeze(1)], 1))
            
            labels_pred = classifier(fake_recons)
            class_acc = class_acc + torch.sum(torch.eq(1 - labels, torch.argmax(labels_pred, 1))).cpu().detach().numpy()

            #fid_tot = fid_tot + get_fid(images, recons, inception) 

    print(class_acc/(100*len(loader)))

def psnr_val(encoder, generator, loader, batch_size):

    psnr = 0

    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            labels = images[1]
            images = images[0]

            images = images.cuda()
            labels = labels.cuda()
            #labels = labels[:, 9].long()

            #Training the discriminator
            lat, cl = encoder(images)
            recons = generator(torch.cat([lat, cl.unsqueeze(1)], 1))

            images = images.detach().cpu().numpy()
            recons = recons.detach().cpu().numpy()
            psnr = psnr + cv2.PSNR(images, recons)

    print(psnr/len(loader))

def plot_sample_images(loader, encoder, generator):

    for batch_idx, images in enumerate(loader):
        labels = images[1]
        images = images[0]

        images = images.cuda()
        labels = labels.cuda()
        labels = labels[:, -1].long()
        #labels = 2*labels - 1
        #labels = labels.reshape(images.size(0), 1)

        #print(labels)

        lat, cl = encoder(images)
        #lat = reparametrize(lat_mu, lat_var)
        recons = generator(torch.cat([lat, cl.unsqueeze(1)], 1))
        recons_fake = generator(torch.cat([lat, 1 - cl.unsqueeze(1)], 1))
        break

    images = images.cpu().detach()
    recons = recons.cpu().detach()
    recons_fake = recons_fake.cpu().detach()
    batch_size = images.size(0)

    #print(torch.mean(torch.abs(images - recons)))
    """
    ls1 = []

    for i in range(0, batch_size):
        r = recons[i]
        r = r - torch.min(r)
        r = r/torch.max(r)
        r = T.ToPILImage()(r)
        ls.append(r)

    plot_im(ls, 'outputs1/mat_out.png')
    """
    ls1 = []

    for i in range(0, batch_size):
        r = recons_fake[i]
        r = r - torch.min(r)
        r = r/torch.max(r)
        r = T.ToPILImage()(r)
        ls1.append(r)

    #plot_im(ls, 'outputs1f/mat_outf.png')
    ls2 = []

    for i in range(0, batch_size):
        r = images[i]
        r = r - torch.min(r)
        r = r/torch.max(r)
        r = T.ToPILImage()(r)
        ls2.append(r)

    plot_im(ls2, ls1, 3, 'comp.png')

def plot_im(images1, images2, n, filename):

    fig = plt.figure(figsize=(4, 6))  # Notice the equal aspect ratio
    plot_per_row = 6
    plot_per_col = 4
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(4*6)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        if i%2==0:
            a.imshow(images1[int(i/2)])
        else:
            a.imshow(images2[int((i-1)/2)])
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename, bbox_inches="tight")


def evaluate_model(dataset_name, encoder, generator, z_dim, pixel, batch_size, device):
    train_loader, test_loader = data_loader(dataset_name, pixel, batch_size)
    plot_sample_images(test_loader, encoder, generator)
    #class_acc(encoder, generator, test_loader, batch_size)
    psnr_val(encoder, generator, test_loader, batch_size)

def discrete_factor_prediction_score(latent_factor_train, ground_truth_train, latent_factor_test, ground_truth_test,
                                     cls):
    if cls == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5)
    elif cls == 'RF':
        model = RandomForestClassifier()
    else:
        model = LinearSVC()

    model.fit(latent_factor_train, ground_truth_train)
    return model.score(latent_factor_test, ground_truth_test)

def continuos_factor_prediction_score(latent_factor_train, ground_truth_train,latent_factor_test,ground_truth_test,cls):
    if cls=='LR':
        model = LinearRegression()
    else:
        model = SVR()
    model.fit(latent_factor_train, ground_truth_train)
    return model.score(latent_factor_test, ground_truth_test)

def calculate_score(dataset_name,model, z_dim, pixel, batch_size, device):
    train_loader, test_loader = data_loader(dataset_name, pixel, batch_size, shuffle=False)
    save_latent_variables(dataset_name, train_loader, model, 'train', pixel, batch_size, device=device)
    save_latent_variables(dataset_name, test_loader, model, 'test', pixel, batch_size, device=device)
    train_phi = np.loadtxt('Harmony_phi_values_'+dataset_name +'_train.np')
    test_phi = np.loadtxt('Harmony_phi_values_' + dataset_name + '_test.np')
    train_semantic_z =  train_phi[:,-z_dim:]
    test_semantic_z = test_phi[:,-z_dim:]
    train_labels = loadpickle('datasets/'+dataset_name+'_train_label.pkl')
    test_labels = loadpickle('datasets/' + dataset_name + '_test_label.pkl')
    train_content = train_labels['Digit']
    test_content = test_labels['Digit']
    return discrete_factor_prediction_score(train_semantic_z.reshape(-1,z_dim),train_content,test_semantic_z.reshape(-1,z_dim),test_content,'svc')

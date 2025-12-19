import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.embedding_quality import embedding_quality
import cne
import numpy as np
import pickle
import torchvision
from sklearn.decomposition import PCA
import pandas as pd
import time

print("Imports completed successfully.")

lambdas_list = [0.0, 0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0]
number_rs = 4

# data
# tasic
tasic_data = np.load('data/tasic/tasic-pca50.npy')
tasic_labels = np.load('data/tasic/tasic-ttypes.npy')
tasic_pca2 = tasic_data[:, :2]
tasic_init = tasic_pca2 / tasic_pca2[:,0].std() * 0.0001

pca = PCA(n_components=2)
tasic_pca2_sk = pca.fit_transform(tasic_data)
tasic_init_weights = pca.components_.T /  tasic_pca2_sk[:,0].std() * 0.0001

# kanton
data_file = "data/Kanton/human-409b2.data.npy"
labels_file = "data/Kanton/human-409b2.labels.npy"
pkl_file = "data/Kanton/human-409b2.pkl"

kanton_data = np.load(data_file)
kanton_labels = np.load(labels_file)
kanton_pca2 = kanton_data[:, :2]
kanton_init = kanton_pca2 / kanton_pca2[:,0].std() * 0.0001

pca = PCA(n_components=2)
kanton_pca2_sk = pca.fit_transform(kanton_data)
kanton_init_weights = pca.components_.T / kanton_pca2_sk[:,0].std() * 0.0001

# genome
genome_data_all = np.loadtxt('data/Genomes/gt_sum_thinned.npy')
genome_data = PCA(n_components=50).fit_transform(genome_data_all)
genome_labels = np.loadtxt('data/Genomes/population_labels.txt', dtype=str)
genome_pca2 = genome_data[:, :2]
genome_init = genome_pca2 / genome_pca2[:,0].std() * 0.0001

pca = PCA(n_components=2)
genome_pca2_sk = pca.fit_transform(genome_data)
genome_init_weights = pca.components_.T / genome_pca2_sk[:,0].std() * 0.0001

# mnist
mnist_train = torchvision.datasets.MNIST(root='data',
                                         train=True,
                                         download=False, 
                                         transform=None)
x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets

mnist_test = torchvision.datasets.MNIST(root='data',
                                        train=False,
                                        download=False, 
                                        transform=None)
x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

pca = PCA(n_components=50)
mnist_data = pca.fit_transform(x_train)
mnist_labels = y_train
mnist_pca2 = mnist_data[:, :2]
mnist_init = mnist_pca2 / mnist_pca2[:,0].std() * 0.0001

pca = PCA(n_components=2)
mnist_pca2_sk = pca.fit_transform(mnist_data)
mnist_init_weights = pca.components_.T / mnist_pca2_sk[:,0].std() * 0.0001

# retina
retina_data = np.load('data/retina/3000_no_std_pca50.npy')
retina_labels = np.load('data/retina/labels 1.npy')
retina_pca2 = retina_data[:, :2]
retina_init = retina_pca2 / retina_pca2[:,0].std() * 0.0001

pca = PCA(n_components=2)
retina_pca2_sk = pca.fit_transform(retina_data)
retina_init_weights = pca.components_.T / retina_pca2_sk[:,0].std() * 0.0001

# Zebrafish
zfish_data = np.load('data/zfish/zfish.data.npy')
zfish_labels = np.load('data/zfish/zfish.labels.npy')
zfish_pca2 = zfish_data[:, :2]
zfish_init = zfish_pca2 / zfish_pca2[:,0].std() * 0.0001

pca = PCA(n_components=2)
zfish_pca2_sk = pca.fit_transform(zfish_data)
zfish_init_weights = pca.components_.T / zfish_pca2_sk[:,0].std() * 0.0001

# C. elegans
c_el_data = np.load('data/c_elegans/c_elegans_50pc.npy')
c_el_labels = np.load('data/c_elegans/c_el_cell_types.npy', allow_pickle=True).astype(str)
c_el_pca2 = c_el_data[:, :2]
c_el_init = c_el_pca2 / c_el_pca2[:,0].std() * 0.0001

c_el_pca2_sk = PCA(n_components=2)
c_el_pca2_sk = c_el_pca2_sk.fit_transform(c_el_data)
c_el_init_weights = pca.components_.T / c_el_pca2_sk[:,0].std() * 0.0001

# mammoth
mammoth_data = np.load('data/mammoth/mammoth_pca.npy')
mammoth_labels = np.load('data/mammoth/mammoth_label.npy')
mammoth_pca = mammoth_data - np.mean(mammoth_data, axis=0)
mammoth_init = mammoth_pca[:, :2] / mammoth_pca[:,0].std() * 0.0001

pca = PCA(n_components=2)
mammoth_pca2_sk = pca.fit_transform(mammoth_data)
mammoth_init_weights = pca.components_.T /  mammoth_pca2_sk[:,0].std() * 0.0001

# fashion MNIST
f_mnist_data = np.load('data/fashion_MNIST/fashion_mnist_pca50.npy')
f_mnist_labels = np.load('data/fashion_MNIST/fashion_mnist_label.npy')
f_mnist_pca = f_mnist_data - np.mean(f_mnist_data, axis=0)
f_mnist_init = f_mnist_pca[:, :2] / f_mnist_pca[:,0].std() * 0.0001

pca = PCA(n_components=2)
f_mnist_pca2_sk = pca.fit_transform(f_mnist_data)
f_mnist_init_weights = pca.components_.T /  f_mnist_pca2_sk[:,0].std() * 0.0001

# satellite
satellite_data = np.load('data/satellite/satellite_pca.npy')
satellite_labels = np.load('data/satellite/satellite_label.npy')
satellite_labels = satellite_labels.ravel()
satellite_pca = satellite_data - np.mean(satellite_data, axis=0)
satellite_init = satellite_pca[:, :2] / satellite_pca[:,0].std() * 0.0001

pca = PCA(n_components=2)
satellite_pca2_sk = pca.fit_transform(satellite_data)
satellite_init_weights = pca.components_.T /  satellite_pca2_sk[:,0].std() * 0.0001

# cifar10
cifar10_data = np.load('data/CIFAR10/cifar10_50pc.npy')
cifar10_labels = np.load('data/CIFAR10/cifar10_labels.npy')
cifar10_pca = cifar10_data - np.mean(cifar10_data, axis=0)
cifar10_init = cifar10_pca[:, :2] / cifar10_pca[:,0].std() * 0.0001

pca = PCA(n_components=2)
cifar10_pca2_sk = pca.fit_transform(cifar10_data)
cifar10_init_weights = pca.components_.T /  cifar10_pca2_sk[:,0].std() * 0.0001


data_list = [
    tasic_data, 
    kanton_data, 
    genome_data, 
    mnist_data, 
    retina_data, 
    zfish_data, 
    c_el_data,
    mammoth_data,
    f_mnist_data,
    satellite_data,
    cifar10_data
]
labels_list = [
    tasic_labels, 
    kanton_labels, 
    genome_labels, 
    mnist_labels, 
    retina_labels, 
    zfish_labels, 
    c_el_labels,
    mammoth_labels,
    f_mnist_labels,
    satellite_labels,
    cifar10_labels
]
init_list = [
    tasic_init, 
    kanton_init, 
    genome_init, 
    mnist_init, 
    retina_init, 
    zfish_init, 
    c_el_init,
    mammoth_init,
    f_mnist_init,
    satellite_init,
    cifar10_init
]
init_weights_list = [
    tasic_init_weights, 
    kanton_init_weights, 
    genome_init_weights, 
    mnist_init_weights, 
    retina_init_weights, 
    zfish_init_weights, 
    c_el_init_weights,
    mammoth_init_weights,
    f_mnist_init_weights,
    satellite_init_weights,
    cifar10_init_weights
]
names_list = [
    "tasic", 
    "kanton", 
    "genome", 
    "MNIST", 
    "retina", 
    "zfish", 
    "c_elegans",
    "Mammoth", 
    "Fashion MNIST", 
    "Satellite", 
    "CIFAR10" 
]

print("Data loaded successfully.")

for k, (data, labels, init, init_weights, name) in enumerate(zip(data_list, labels_list, init_list, init_weights_list, names_list)):
    print(f'------------------------- {name} -------------------------')
    
    results_dict = {}
    results_dict_dec = {}

    if name == 'Satellite' or name == 'kanton':
        classes = 4
    else:
        classes = 6
    for seed in range(number_rs):
        seed_key = f"seed_{seed}"
        results_dict[seed_key] = {}
        results_dict_dec[seed_key] = {}

        for i, l in enumerate(lambdas_list):
            print(f'Dataset: {name} - Running {i}/{len(lambdas_list)*number_rs} with lambda {l} and seed {seed}')

            l_key = f"lambda_{l}"
            
            # Regularizer
            start = time.perf_counter()
            embedder = cne.CNE(seed=i, negative_samples=500, regularizer=True, reg_embedding=init, reg_lambda=l, reg_scaling='norm', n_epochs=750)
            embd = embedder.fit_transform(data)
            end = time.perf_counter()
            eval = embedding_quality(embd, data, labels, seed=seed, knn_classes=classes)

            results_dict[seed_key][l_key] = {
                'embedding': np.array(embd),
                'eval': eval,
                'time': end - start
            }

            # Decoder
            start = time.perf_counter()
            embedder_dec = cne.CNE(seed=i, negative_samples=500, decoder=True, reg_lambda=l, n_epochs=750)
            embd_dec, weights = embedder_dec.fit_transform(data, init_weights=init_weights)
            end = time.perf_counter()
            eval_dec = embedding_quality(embd_dec, data, labels, seed=seed, knn_classes=classes)
            results_dict_dec[seed_key][l_key] = {
                'embedding': np.array(embd_dec),
                'eval': eval_dec,
                'time': end - start
            }
    os.makedirs('results/dreams/dreams_cne', exist_ok=True)

    with open(f'results/dreams/dreams_cne/{name}_results_dreams_cne.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

    with open(f'results/dreams/dreams_cne/{name}_results_dreams_cne_dec.pkl', 'wb') as f:
        pickle.dump(results_dict_dec, f)
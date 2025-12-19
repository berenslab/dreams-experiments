import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.embedding_quality import embedding_quality
import numpy as np
import pickle
import openTSNE
from openTSNE import TSNE
import torchvision
from sklearn.decomposition import PCA
import pandas as pd
import time

print("Imports completed successfully.")

lambdas_list = np.linspace(0, 1, 41)
number_rs = 4

# Load om data
with open('results/tasic_results_om.pkl', 'rb') as f:
    tasic_om_results = pickle.load(f)

with open('results/kanton_results_om.pkl', 'rb') as f:
    kanton_om_results = pickle.load(f)

with open('results/genome_results_om.pkl', 'rb') as f:
    genome_om_results = pickle.load(f)

with open('results/MNIST_results_om.pkl', 'rb') as f:
    mnist_om_results = pickle.load(f)

with open('results/retina_results_om.pkl', 'rb') as f:
    retina_om_results = pickle.load(f)

with open('results/zfish_results_om.pkl', 'rb') as f:
    zfish_om_results = pickle.load(f)

with open('results/c_elegans_results_om.pkl', 'rb') as f:
    c_el_om_results = pickle.load(f)

with open('results/mammoth/mammoth_results.pkl', 'rb') as f:
    mammoth_om_results = pickle.load(f)

with open('results/fashion_mnist/fashion_mnist_results.pkl', 'rb') as f:
    f_mnist_om_results = pickle.load(f)

with open('results/satellite/satellite_results.pkl', 'rb') as f:
    satellite_om_results = pickle.load(f)

with open('results/satellite/satellite_results.pkl', 'rb') as f:
    satellite_om_results = pickle.load(f)

with open('results/cifar10/cifar10_results.pkl', 'rb') as f:
    cifar10_om_results = pickle.load(f)

# data
# tasic
tasic_data = np.load('data/tasic/tasic-pca50.npy')
tasic_labels = np.load('data/tasic/tasic-ttypes.npy')
tasic_mds = tasic_om_results['squad_mds']['seed_0']['embedding']
tasic_mds = tasic_mds - tasic_mds.mean(axis=0)
tasic_init = tasic_mds / tasic_mds[:,0].std() * 0.0001

# kanton
data_file = "data/Kanton/human-409b2.data.npy"
labels_file = "data/Kanton/human-409b2.labels.npy"
pkl_file = "data/Kanton/human-409b2.pkl"

kanton_data = np.load(data_file)
kanton_labels = np.load(labels_file)
kanton_mds = kanton_om_results['squad_mds']['seed_0']['embedding']
kanton_mds = kanton_mds - kanton_mds.mean(axis=0)
kanton_init = kanton_mds / kanton_mds[:,0].std() * 0.0001

# genome
genome_data_all = np.loadtxt('data/Genomes/gt_sum_thinned.npy')
genome_data = PCA(n_components=50).fit_transform(genome_data_all)
genome_labels = np.loadtxt('data/Genomes/population_labels.txt', dtype=str)
genome_mds = genome_om_results['squad_mds']['seed_0']['embedding']
genome_mds = genome_mds - genome_mds.mean(axis=0)
genome_init = genome_mds / genome_mds[:,0].std() * 0.0001

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
mnist_mds = mnist_om_results['squad_mds']['seed_0']['embedding']
mnist_mds = mnist_mds - mnist_mds.mean(axis=0)
mnist_init = mnist_mds / mnist_mds[:,0].std() * 0.0001

# retina
retina_data = np.load('data/retina/3000_no_std_pca50.npy')
retina_labels = np.load('data/retina/labels 1.npy')
retina_mds = retina_om_results['squad_mds']['seed_0']['embedding']
retina_mds = retina_mds - retina_mds.mean(axis=0)
retina_init = retina_mds / retina_mds[:,0].std() * 0.0001

# Zebrafish
zfish_data = np.load('data/zfish/zfish.data.npy')
zfish_labels = np.load('data/zfish/zfish.labels.npy')
zfish_mds = zfish_om_results['squad_mds']['seed_0']['embedding']
zfish_mds = zfish_mds - zfish_mds.mean(axis=0)
zfish_init = zfish_mds / zfish_mds[:,0].std() * 0.0001

# C. elegans
c_el_data = np.load('data/c_elegans/c_elegans_50pc.npy')
c_el_labels = np.load('data/c_elegans/c_el_cell_types.npy', allow_pickle=True).astype(str)
c_el_mds = c_el_om_results['squad_mds']['seed_0']['embedding']
c_el_mds = c_el_mds - c_el_mds.mean(axis=0)
c_el_init = c_el_mds / c_el_mds[:,0].std() * 0.0001

# mammoth
mammoth = np.load('data/mammoth/mammoth_pca.npy')
mammoth_labels = np.load('data/mammoth/mammoth_label.npy')
mammoth_mds = mammoth_om_results[0]['squad_mds']['embedding']
mammoth_mds = mammoth_mds - mammoth_mds.mean(axis=0)
mammoth_init = mammoth_mds / mammoth_mds[:,0].std() * 0.0001

# fashion_mnist
f_mnist = np.load('data/fashion_MNIST/fashion_mnist_pca50.npy')
f_mnist_labels = np.load('data/fashion_MNIST/fashion_mnist_label.npy')
f_mnist_mds = f_mnist_om_results[0]['squad_mds']['embedding']
f_mnist_mds = f_mnist_mds - f_mnist_mds.mean(axis=0)
f_mnist_init = f_mnist_mds / f_mnist_mds[:,0].std() * 0.0001

# satellite
satellite = np.load('data/satellite/satellite_pca.npy')
satellite_labels = np.load('data/satellite/satellite_label.npy')
satellite_labels = satellite_labels.ravel()
satellite_mds = satellite_om_results[0]['squad_mds']['embedding']
satellite_mds = satellite_mds - satellite_mds.mean(axis=0)
satellite_init = satellite_mds / satellite_mds[:,0].std() * 0.0001

# cifar10
cifar10 = np.load('data/CIFAR10/cifar10_50pc.npy')
cifar10_labels = np.load('data/CIFAR10/cifar10_labels.npy')
cifar10_mds = cifar10_om_results[0]['squad_mds']['embedding']
cifar10_mds = cifar10_mds - cifar10_mds.mean(axis=0)
cifar10_init = cifar10_mds / cifar10_mds[:,0].std() * 0.0001

data_list = [
    tasic_data, 
    kanton_data, 
    genome_data, 
    mnist_data, 
    retina_data, 
    zfish_data, 
    c_el_data,
    mammoth, 
    f_mnist, 
    satellite, 
    cifar10
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

for data, labels, init, name in zip(data_list, labels_list, init_list, names_list):
    print(f'------------------------- {name} -------------------------')
    
    results_dict = {}
    for seed in range(number_rs):
        seed_key = f"seed_{seed}"
        results_dict[seed_key] = {}

        for i, l in enumerate(lambdas_list):
            print(f'Running {(i+1)+(seed*len(lambdas_list))}/{len(lambdas_list)*number_rs} with lambda {l}')
            start = time.perf_counter()
            embedder = TSNE(initialization=init, regularization=True, reg_lambda=l, reg_embedding=init, reg_scaling='norm', reg_scaling_dims='one', random_state=seed)
            embd = embedder.fit(data)
            end = time.perf_counter()
            if name == 'Satellite' or name == 'kanton':
                classes = 4
            else:
                classes = 6
            eval = embedding_quality(embd, data, labels, seed=seed, knn_classes=classes)

            l_key = f"lambda_{l}"
            results_dict[seed_key][l_key] = {
                'embedding': np.array(embd),
                'eval': eval,
                'time': end - start
            }
    os.makedirs('results/dreams/dreams_mds', exist_ok=True)

    with open(f'results/dreams/dreams_mds/{name}_results_dreams_mds.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
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

print("Imports completed successfully.")

lambdas_list = [0.0, 0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0]
number_rs = 4

# data
# tasic
tasic_data = np.load('data/tasic/tasic-pca50.npy')
tasic_labels = np.load('data/tasic/tasic-ttypes.npy')
tasic_pca2 = tasic_data[:, :2]
tasic_init = tasic_pca2 / tasic_pca2[:,0].std()

pca = PCA(n_components=2)
tasic_pca2_sk = pca.fit_transform(tasic_data)
tasic_init_weights = pca.components_.T /  tasic_pca2_sk[:,0].std()

# kanton
data_file = "data/Kanton/human-409b2.data.npy"
labels_file = "data/Kanton/human-409b2.labels.npy"
pkl_file = "data/Kanton/human-409b2.pkl"

kanton_data = np.load(data_file)
kanton_labels = np.load(labels_file)
kanton_pca2 = kanton_data[:, :2]
kanton_init = kanton_pca2 / kanton_pca2[:,0].std()

pca = PCA(n_components=2)
kanton_pca2_sk = pca.fit_transform(kanton_data)
kanton_init_weights = pca.components_.T / kanton_pca2_sk[:,0].std()

# genome
genome_data_all = np.loadtxt('data/Genomes/gt_sum_thinned.npy')
genome_data = PCA(n_components=50).fit_transform(genome_data_all)
genome_labels = np.loadtxt('data/Genomes/population_labels.txt', dtype=str)
genome_pca2 = genome_data[:, :2]
genome_init = genome_pca2 / genome_pca2[:,0].std()

pca = PCA(n_components=2)
genome_pca2_sk = pca.fit_transform(genome_data)
genome_init_weights = pca.components_.T / genome_pca2_sk[:,0].std()

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
mnist_init = mnist_pca2 / mnist_pca2[:,0].std()

pca = PCA(n_components=2)
mnist_pca2_sk = pca.fit_transform(mnist_data)
mnist_init_weights = pca.components_.T / mnist_pca2_sk[:,0].std()

# retina
retina_data = np.load('data/retina/3000_no_std_pca50.npy')
retina_labels = np.load('data/retina/labels 1.npy')
retina_pca2 = retina_data[:, :2]
retina_init = retina_pca2 / retina_pca2[:,0].std()

pca = PCA(n_components=2)
retina_pca2_sk = pca.fit_transform(retina_data)
retina_init_weights = pca.components_.T / retina_pca2_sk[:,0].std()

# Zebrafish
zfish_data = np.load('data/zfish/zfish.data.npy')
zfish_labels = np.load('data/zfish/zfish.labels.npy')
zfish_pca2 = zfish_data[:, :2]
zfish_init = zfish_pca2 / zfish_pca2[:,0].std()

pca = PCA(n_components=2)
zfish_pca2_sk = pca.fit_transform(zfish_data)
zfish_init_weights = pca.components_.T / zfish_pca2_sk[:,0].std()

# C. elegans
c_el_data = np.load('data/c_elegans/c_elegans_50pc.npy')
c_el_labels = np.load('data/c_elegans/c_el_cell_types.npy', allow_pickle=True).astype(str)
c_el_pca2 = c_el_data[:, :2]
c_el_init = c_el_pca2 / c_el_pca2[:,0].std()

c_el_pca2_sk = PCA(n_components=2)
c_el_pca2_sk = c_el_pca2_sk.fit_transform(c_el_data)
c_el_init_weights = pca.components_.T / c_el_pca2_sk[:,0].std()

data_list = [
    tasic_data, 
    kanton_data, 
    genome_data, 
    mnist_data, 
    retina_data, 
    zfish_data, 
    c_el_data
]
labels_list = [
    tasic_labels, 
    kanton_labels, 
    genome_labels, 
    mnist_labels, 
    retina_labels, 
    zfish_labels, 
    c_el_labels
]
init_list = [
    tasic_init, 
    kanton_init, 
    genome_init, 
    mnist_init, 
    retina_init, 
    zfish_init, 
    c_el_init
]
init_weights_list = [
    tasic_init_weights, 
    kanton_init_weights, 
    genome_init_weights, 
    mnist_init_weights, 
    retina_init_weights, 
    zfish_init_weights, 
    c_el_init_weights
]
names_list = [
    "tasic", 
    "kanton", 
    "genome", 
    "MNIST", 
    "retina", 
    "zfish", 
    "c_elegans"
]

print("Data loaded successfully.")

for k, (data, labels, init, init_weights, name) in enumerate(zip(data_list, labels_list, init_list, init_weights_list, names_list)):
    print(f'------------------------- {name} -------------------------')
    
    results_dict = {}
    results_dict_dec = {}
    for seed in range(number_rs):
        seed_key = f"seed_{seed}"
        results_dict[seed_key] = {}
        results_dict_dec[seed_key] = {}

        for i, l in enumerate(lambdas_list):
            print(f'Dataset: {name} - Running {i}/{len(lambdas_list)*number_rs} with lambda {l} and seed {seed}')

            l_key = f"lambda_{l}"
            
            # Regularizer
            embedder = cne.CNE(seed=i, negative_samples=500, regularizer=True, reg_embedding=init, reg_lambda=l, reg_scaling='norm', n_epochs=750)
            embd = embedder.fit_transform(data)
            eval = embedding_quality(embd, data, labels, seed=seed)

            results_dict[seed_key][l_key] = {
                'embedding': np.array(embd),
                'eval': eval
            }

            # Decoder
            embedder_dec = cne.CNE(seed=i, negative_samples=500, decoder=True, reg_lambda=l, n_epochs=750)
            embd_dec, weights = embedder_dec.fit_transform(data, init_weights=init_weights)
            eval_dec = embedding_quality(embd_dec, data, labels, seed=seed)
            results_dict_dec[seed_key][l_key] = {
                'embedding': np.array(embd_dec),
                'eval': eval_dec
            }
    os.makedirs('results/dreams/dreams_cne', exist_ok=True)

    with open(f'results/dreams/dreams_cne/{name}_results_dreams_cne_big.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

    with open(f'results/dreams/dreams_cne/{name}_results_dreams_cne_dec_big.pkl', 'wb') as f:
        pickle.dump(results_dict_dec, f)
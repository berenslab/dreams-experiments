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

# data
# tasic
tasic_data = np.load('data/tasic/tasic-pca50.npy')
tasic_labels = np.load('data/tasic/tasic-ttypes.npy')
tasic_init = tasic_om_results['squad_mds']['seed_0']['embedding']

# kanton
data_file = "data/Kanton/human-409b2.data.npy"
labels_file = "data/Kanton/human-409b2.labels.npy"
pkl_file = "data/Kanton/human-409b2.pkl"

kanton_data = np.load(data_file)
kanton_labels = np.load(labels_file)
kanton_init = kanton_om_results['squad_mds']['seed_0']['embedding']

# genome
genome_data_all = np.loadtxt('data/Genomes/gt_sum_thinned.npy')
genome_data = PCA(n_components=50).fit_transform(genome_data_all)
genome_labels = np.loadtxt('data/Genomes/population_labels.txt', dtype=str)
genome_init = genome_om_results['squad_mds']['seed_0']['embedding']

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
mnist_init = mnist_om_results['squad_mds']['seed_0']['embedding']

# retina
retina_data = np.load('data/retina/3000_no_std_pca50.npy')
retina_labels = np.load('data/retina/labels 1.npy')
retina_init = retina_om_results['squad_mds']['seed_0']['embedding']

# Zebrafish
zfish_data = np.load('data/zfish/zfish.data.npy')
zfish_labels = np.load('data/zfish/zfish.labels.npy')
zfish_init = zfish_om_results['squad_mds']['seed_0']['embedding']

# C. elegans
c_el_data = np.load('data/c_elegans/c_elegans_50pc.npy')
c_el_labels = np.load('data/c_elegans/c_el_cell_types.npy', allow_pickle=True).astype(str)
c_el_init = c_el_om_results['squad_mds']['seed_0']['embedding']

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

for data, labels, init, name in zip(data_list, labels_list, init_list, names_list):
    print(f'------------------------- {name} -------------------------')
    
    results_dict = {}
    for seed in range(number_rs):
        seed_key = f"seed_{seed}"
        results_dict[seed_key] = {}

        for i, l in enumerate(lambdas_list):
            print(f'Running {i+1}/{len(lambdas_list)*number_rs} with lambda {l}')
            embedder = TSNE(initialization=init, regularization=True, reg_lambda=l, reg_embedding=init, reg_scaling='norm', reg_scaling_dims='one', random_state=seed)
            embd = embedder.fit(data)
            eval = embedding_quality(embd, data, labels, seed=seed)

            l_key = f"lambda_{l}"
            results_dict[seed_key][l_key] = {
                'embedding': np.array(embd),
                'eval': eval
            }
    os.makedirs('results/dreams/dreams_mds', exist_ok=True)

    with open(f'results/dreams/dreams_mds/{name}_results_dreams_mds.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.embedding_quality import embedding_quality
import numpy as np
import pickle
import torchvision
from sklearn.decomposition import PCA
import pandas as pd

print("Imports completed successfully.")

# data
# tasic
tasic_data = np.load('data/tasic/tasic-pca50.npy')
tasic_labels = np.load('data/tasic/tasic-ttypes.npy')
tasic_pca2 = tasic_data[:, :2]
tasic_init = tasic_pca2 / tasic_pca2[:,0].std()

# kanton
data_file = "data/Kanton/human-409b2.data.npy"
labels_file = "data/Kanton/human-409b2.labels.npy"
pkl_file = "data/Kanton/human-409b2.pkl"

kanton_data = np.load(data_file)
kanton_labels = np.load(labels_file)
kanton_pca2 = kanton_data[:, :2]
kanton_init = kanton_pca2 / kanton_pca2[:,0].std()

# genome
genome_data_all = np.loadtxt('data/Genomes/gt_sum_thinned.npy')
genome_data = PCA(n_components=50).fit_transform(genome_data_all)
genome_labels = np.loadtxt('data/Genomes/population_labels.txt', dtype=str)
genome_pca2 = genome_data[:, :2]
genome_init = genome_pca2 / genome_pca2[:,0].std()

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

# retina
retina_data = np.load('data/retina/3000_no_std_pca50.npy')
retina_labels = np.load('data/retina/labels 1.npy')
retina_pca2 = retina_data[:, :2]
retina_init = retina_pca2 / retina_pca2[:,0].std()

# Zebrafish
zfish_data = np.load('data/zfish/zfish.data.npy')
zfish_labels = np.load('data/zfish/zfish.labels.npy')
zfish_pca2 = zfish_data[:, :2]
zfish_init = zfish_pca2 / zfish_pca2[:,0].std()

# C. elegans
c_el_data = np.load('data/c_elegans/c_elegans_50pc.npy')
c_el_labels = np.load('data/c_elegans/c_el_cell_types.npy', allow_pickle=True).astype(str)
c_el_pca2 = c_el_data[:, :2]
c_el_init = c_el_pca2 / c_el_pca2[:,0].std()

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


from squad_mds.SQuaD_MDS import run_SQuaD_MDS
from squad_mds.hybrid import run_hybrid
import pacmap
import phate
import trimap
from starmap.star_umap import StarMAP, StarMAP_v2
from starmap.umap_base import UMAP_base
import umap

number_rs = 4

print("Starting the embedding process...")

for i in range(len(data_list)):
    data_name = names_list[i]
    data = data_list[i]
    labels = labels_list[i]
    init = init_list[i]

    results_mds = {}
    results_mds_hybrid = {}
    results_pacmap = {}
    results_phate = {}
    results_trimap = {}
    results_star_map = {}
    results_umap = {}

    for rs in range(number_rs):
        print(f"Running {data_name} with random seed {rs}")
        seed_key = f"seed_{rs}"

        # SquAD-MDS
        embd = run_SQuaD_MDS(data, {'in python':False})
        eval = embedding_quality(embd, data, labels, seed=rs)

        results_mds[seed_key] = {
            'embedding': embd,
            'eval': eval,
        }
        print(f"Squad MDS - {data_name} - Seed {rs} completed.")

        embd = run_hybrid(data, {'in python': True})
        eval = embedding_quality(embd, data, labels, seed=rs)
        results_mds_hybrid[seed_key] = {
            'embedding': embd,
            'eval': eval,
        }
        print(f"Squad MDS Hybrid - {data_name} - Seed {rs} completed.")

        # pacmap
        embedder = pacmap.PaCMAP(n_components=2, random_state=rs)
        embd = embedder.fit_transform(data, init='pca')
        eval = embedding_quality(embd, data, labels, seed=rs)

        results_pacmap[seed_key] = {
            'embedding': embd,
            'eval': eval,
        }
        print(f"PaCMAP - {data_name} - Seed {rs} completed.")

        # phate
        embedder = phate.PHATE(n_jobs=-2, random_state=rs)
        embd = embedder.fit_transform(data)
        eval = embedding_quality(embd, data, labels, seed=rs)

        results_phate[seed_key] = {
            'embedding': embd,
            'eval': eval,
        }
        print(f"PHATE - {data_name} - Seed {rs} completed.")

        # trimap
        embd = trimap.TRIMAP().fit_transform(data)
        eval = embedding_quality(embd, data, labels, seed=rs)

        results_trimap[seed_key] = {
            'embedding': embd,
            'eval': eval,
        }
        print(f"TRIMAP - {data_name} - Seed {rs} completed.")

        # starmap
        embedder = StarMAP_v2(data, n_clusters=75, random_state=rs)
        embedder.optimize()
        embd = embedder.embedding

        results_star_map[seed_key] = {
            'embedding': embd,
            'eval': embedding_quality(embd, data, labels, seed=rs),
        }
        print(f"Star-MAP - {data_name} - Seed {rs} completed.")

        # umap
        embedder = umap.UMAP(n_components=2, random_state=rs)
        embd = embedder.fit_transform(data)
        eval = embedding_quality(embd, data, labels, seed=rs)

        results_umap[seed_key] = {
            'embedding': embd,
            'eval': eval,
        }
        print(f"UMAP - {data_name} - Seed {rs} completed.")

    # Save results (for one dataset)
    os.makedirs('results/other_methods', exist_ok=True)

    with open(f'results/other_methods/{data_name}_results_om_final.pkl', 'wb') as f:
        pickle.dump({
            'squad_mds': results_mds,
            'squad_mds_hybrid': results_mds_hybrid,
            'pacmap': results_pacmap,
            'phate': results_phate,
            'trimap': results_trimap,
            'star_map': results_star_map,
            'umap': results_umap
        }, f)

    print(f"Results for {data_name} saved.")









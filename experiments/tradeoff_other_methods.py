import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.embedding_quality import embedding_quality
import numpy as np
import pickle
import pandas as pd

from squad_mds.SQuaD_MDS import run_SQuaD_MDS
from squad_mds.hybrid import run_hybrid

from starmap.star_umap import StarMAP, StarMAP_v2
from starmap.umap_base import UMAP_base

# data
data = np.load('data/tasic/tasic-pca50.npy')
labels = np.load('data/tasic/tasic-ttypes.npy')

# MDS hybrid
lr_tsne = np.linspace(0, 1.5, 43)
lr_mds = 1.5 - lr_tsne

results_mds = {}

for seed in range(4):
    seed_key = f"seed_{seed}"
    results_mds[seed_key] = {}
    for i, l in enumerate(lr_mds):
        print(f'Running tasic ({(seed+1)*i / len(lr_mds)*4})')
        embd = run_hybrid(data, {'in python': True, 'MDS LR': lr_mds[i], 'tSNE LR': lr_tsne[i]}, random_state=seed)
        eval = embedding_quality(embd, data, labels, seed=seed)

        lam_key = f"lambda_{l}"

        results_mds[seed_key][lam_key] = {
            "embedding": embd,
            "eval": eval,
            "lr_mds/tsne": np.array([lr_mds[i], lr_tsne[i]]),
        }

os.makedirs('results/other_methods', exist_ok=True)

with open('results/other_methods/tasic_mds_hybrid_curve.pkl', 'wb') as f:
    pickle.dump(results_mds, f)

# StarMAP
lambdas = np.linspace(start=0, stop=1, num=41)

results_starmap = {}
for i in range(4):
    seed_key = f"seed_{i}"
    results_starmap[seed_key] = {}
    for l in lambdas:
        embedder = StarMAP_v2(data, n_clusters=75, blend_ratio=l, random_state=i)
        embedder.optimize()
        embd = embedder.embedding
        eval = embedding_quality(embd, data, labels, seed=i)

        lam_key = f"lambda_{l}"

        results_starmap[seed_key][lam_key] = {
            "embedding": embd,
            "eval": eval,
        }

with open('results/other_methods/tasic_starmap_curve_mrs', 'wb') as f:
    pickle.dump(results_starmap, f)


# MDS C-el
# C. elegans
data = np.load('data/c_elegans/c_elegans_50pc.npy')
labels = np.load('data/c_elegans/c_el_cell_types.npy', allow_pickle=True).astype(str)

lr_tsne = np.linspace(0, 1.5, 43)
lr_mds = 1.5 - lr_tsne

results_mds = {}

for seed in range(4):
    seed_key = f"seed_{seed}"
    results_mds[seed_key] = {}
    for i, l in enumerate(lr_mds):
        print(f'Running C-elegans ({(seed+1)*i / len(lr_mds)*4})')

        embd = run_hybrid(data, {'in python': True, 'MDS LR': lr_mds[i], 'tSNE LR': lr_tsne[i]}, random_state=seed)
        eval = embedding_quality(embd, data, labels, seed=seed)

        lam_key = f"lambda_{l}"

        results_mds[seed_key][lam_key] = {
            "embedding": embd,
            "eval": eval,
            "lr_mds/tsne": np.array([lr_mds[i], lr_tsne[i]]),
        }

with open('results/other_methods/c_elegans_mds_hybrid_curve.pkl', 'wb') as f:
    pickle.dump(results_mds, f)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.embedding_quality import embedding_quality
import openTSNE
from openTSNE import TSNE
import pickle
import numpy as np
import os

print("Imports completed successfully.")

tasic_pca50 = np.load('data/tasic/tasic-pca50.npy')
tasic_ttypes = np.load('data/tasic/tasic-ttypes.npy')

tasic_pca2 = tasic_pca50[:, :2]
tasic_pca2_scaled = tasic_pca2 / tasic_pca2[:,0].std()

exag_vals = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 15.0, 18.5, 22.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 350.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0]

results_dict = {}
for seed in range(1):
    seed_key = f"seed_{seed}"
    results_dict[seed_key] = {}

    for i, exag in enumerate(exag_vals):
        print(f'Running {i+1}/{len(exag_vals)}')
        embedder = TSNE(initialization=tasic_pca2_scaled, exaggeration=exag)
        embd = embedder.fit(tasic_pca50)
        eval = embedding_quality(embd, tasic_pca50, tasic_ttypes)

        exag_key = f"exag_{exag}"
        results_dict[seed_key][exag_key] = {
            'embedding': np.array(embd),
            'eval': eval
        }

os.makedirs('results/other_methods', exist_ok=True)

with open('results/other_methods/tasic_results_opentsne_exaggeration.pkl', 'wb') as f:
    pickle.dump(results_dict, f)

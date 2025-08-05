DREAMS - Experiments
========

This repossitory contains the code to reproduce the experiments of "DREAMS: Preserving both Local and Global Structure in Dimensionality Reduction".

DREAMS combines the local structure preservation of $t$-SNE with the global structure preservation of PCA via a regularization term that motivates global structure preservation. It provides a continuum of embeddings along a local-global spectrum with almost no local/global structure preservation tradeoff.

<img width="800" alt="Example DREAMS" src="figures/dreams_spectrum.png">

The code depends on several repositories, especially on [openTSNE](https://github.com/pavlin-policar/openTSNE) and [contrastive-ne](https://github.com/berenslab/contrastive-ne), which build the $t$-SNE backend that DREAMS builds upon.

# Installation

Clone this repository:
```
git clone https://github.com/berenslab/dreams-experiments.git
cd dreams-experiments
```
Create a conda environment (we used Python 3.12.11):
````
conda create -n myenv python=3.12
conda activate myenv
pip install -r requirements.txt
````
Install openTSNE with regularizer (DREAMS):
````
git clone --branch tp --single-branch https://github.com/NavidadK/DREAMS.git
cd openTSNE
python setup.py install
cd ..
````
and contrastive-ne with a regularizer (DREAMS-CNE)
````
git clone --branch tp --single-branch https://github.com/NavidadK/DREAMS-CNE.git
cd contrastive-ne
pip install --no-deps .
````
To perform the experiments, the data sets mentioned in the paper need to be downloaded and preprocessed as described in the paper and saved in the folder /data. The preprocessed [Tasic et al. (2018)](https://www.nature.com/articles/s41586-018-0654-5) can already be found there.

To run experiments that entail methods that we compare against, first clone the [SQuadMDS](https://github.com/PierreLambert3/SQuaD-MDS-and-FItSNE-hybrid) and then run (Note: in the code we also compare against [StarMAP](https://arxiv.org/abs/2502.03776), whose code is as of this moment not publicly available):
````
python experiments/run_other_methods.py
python experiments/run_openTSNE_exag.py
````
To run the DREAMS experiments (run_dreams_mds is using the MDS embedding from run_other_methods.py):
````
python experiments/run_dreams_pca.py
python experiments/run_dreams_mds.py
python experiments/run_dreams_cne.py
python experiments/tradeoff_other_methods.py
````
To analyze and plot the results use:
````
experiments/plot_results_paper.ipynb
````
Here all plots of the paper can be found.

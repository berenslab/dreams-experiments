import numpy as np
import pandas as pd
from IPython.display import display

def analyze_rs(data_dict, 
               display_df=True, 
               find_best=True, 
               opt_knn=None, 
               worst_knn=None, 
               opt_cpd=None, 
               worst_cpd=None, 
               score_type='relative_score'):

    # KNN and CPD dfs
    dict_knn_scores = {}
    dict_cpd_scores = {}

    for seed_key, lambda_dict in data_dict.items():
        dict_knn_scores[seed_key] = {
        lambda_key: lambda_data["eval"][0] for lambda_key, lambda_data in lambda_dict.items()
    }

    df_knn_scores = pd.DataFrame.from_dict(dict_knn_scores, orient='index')
    if display_df:
        print("KNN Scores DataFrame:")
        display(df_knn_scores)

    for seed_key, lambda_dict in data_dict.items():
        dict_cpd_scores[seed_key] = {
        lambda_key: lambda_data["eval"][2] for lambda_key, lambda_data in lambda_dict.items()
    }

    df_cpd_scores = pd.DataFrame.from_dict(dict_cpd_scores, orient='index')
    if display_df:
        print("CPD Scores DataFrame:")
        display(df_cpd_scores)
    
    # Means and stds (for plotting)
    knn_means = df_knn_scores.mean(axis=0)
    knn_stds = df_knn_scores.std(axis=0)
    cpd_means = df_cpd_scores.mean(axis=0)
    cpd_stds = df_cpd_scores.std(axis=0)

    # Find best performing embedding
    if find_best:

        # compute score
        n_seeds, n_lambdas = df_knn_scores.shape
        scores = np.zeros((n_seeds, n_lambdas))

         # relative to worst scores
        for i in range(n_seeds):
            for j in range(n_lambdas):
                knn = df_knn_scores.iloc[i, j]
                cpd = df_cpd_scores.iloc[i, j]

                if opt_knn is None:
                    opt_knn = knn_means.max()
                if worst_knn is None:
                    worst_knn = knn_means.min()
                if opt_cpd is None:
                    opt_cpd = cpd_means.max()
                if worst_cpd is None:
                    worst_cpd = cpd_means.min()

                if score_type == 'relative_score': # relative to worst scores
                    scores[i, j] = 1/2 * ( (knn - worst_knn) / (opt_knn - worst_knn) + (cpd - worst_cpd) / (opt_cpd - worst_cpd) )
                
                elif score_type == 'zero_score': # score relative to zero
                    scores[i, j] = 1/2 * ( knn / opt_knn + cpd / opt_cpd )
            
        # best individual score
        max_idx_flat = np.argmax(scores)
        max_idx_2d = np.unravel_index(max_idx_flat, scores.shape)
        max_seed = df_knn_scores.index[max_idx_2d[0]]
        max_lambda = df_knn_scores.columns[max_idx_2d[1]]
        max_score = scores[max_idx_2d[0], max_idx_2d[1]]

        # best score over seeds
        mean_scores = scores.mean(axis=0)
        max_mean_scores_idx = np.argmax(mean_scores)
        max_mean_scores_lambda = df_knn_scores.columns[max_mean_scores_idx]
        
        # check if best lambda is the same as best mean lambda
        if max_lambda != max_mean_scores_lambda:
            print(f"Warning: Best lambda ({max_lambda}) is not the same as best mean lambda ({max_mean_scores_lambda}).")

        # return best embedding and evaluation
        best_embedding = data_dict[max_seed][max_lambda]['embedding']
        best_embedding_eval = data_dict[max_seed][max_lambda]['eval']

        return df_knn_scores, df_cpd_scores, knn_means, knn_stds, cpd_means, cpd_stds, scores, mean_scores, max_mean_scores_idx, max_mean_scores_lambda, max_seed, max_lambda, max_score, best_embedding, best_embedding_eval
    else:
        return df_knn_scores, df_cpd_scores, knn_means, knn_stds, cpd_means, cpd_stds


def analyze_rs_om(data_dict, display_df=True):
    methods = list(data_dict.keys())
    all_seeds = list(data_dict[methods[0]].keys())

    df_knn = pd.DataFrame(index=all_seeds, columns=methods)
    df_cpd = pd.DataFrame(index=all_seeds, columns=methods)

    mean_knn_all = []
    mean_cpd_all = []
    std_knn_all = []
    std_cpd_all = []

    for method in methods:
        values_0 = []
        values_2 = []
        for seed in all_seeds:
            if seed in data_dict[method]:
                val_0 = data_dict[method][seed]['eval'][0]
                val_2 = data_dict[method][seed]['eval'][2]
                df_knn.loc[seed, method] = val_0
                df_cpd.loc[seed, method] = val_2
                values_0.append(val_0)
                values_2.append(val_2)
        mean_knn = sum(values_0) / len(values_0) if values_0 else None
        mean_knn_all.append(mean_knn)
        mean_cpd = sum(values_2) / len(values_2) if values_2 else None
        mean_cpd_all.append(mean_cpd)
        std_knn = np.std(values_0, ddof=1) if len(values_0) > 1 else None
        std_knn_all.append(std_knn)
        std_cpd = np.std(values_2, ddof=1) if len(values_2) > 1 else None
        std_cpd_all.append(std_cpd)

    mean_knn_all = np.array(mean_knn_all)
    mean_cpd_all = np.array(mean_cpd_all)
    std_knn_all = np.array(std_knn_all)
    std_cpd_all = np.array(std_cpd_all)

    if display_df:
        print("KNN Scores DataFrame:")
        display(df_knn)
        print("CPD Scores DataFrame:")
        display(df_cpd)
    
    dict_eval = {}
    for i, method in enumerate(methods):
        dict_eval[method] = {
            'knn_mean': mean_knn_all[i],
            'knn_std': std_knn_all[i],
            'cpd_mean': mean_cpd_all[i],
            'cpd_std': std_cpd_all[i]
        }
        # dict_knn[method] = mean_knn_all[i]
        # dict_cpd[method] = mean_cpd_all[i]

    return df_knn, df_cpd, mean_knn_all, mean_cpd_all, std_knn_all, std_cpd_all, methods, dict_eval  

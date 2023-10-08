import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

import networkx as nx
import scipy

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

def read_files():
    train_a = pd.read_parquet('../A/train_targets.parquet')
    train_b = pd.read_parquet('../B/train_targets.parquet')
    train_c = pd.read_parquet('../C/train_targets.parquet')

    X_train_estimated_a = pd.read_parquet('../A/X_train_estimated.parquet')
    X_train_estimated_b = pd.read_parquet('../B/X_train_estimated.parquet')
    X_train_estimated_c = pd.read_parquet('../C/X_train_estimated.parquet')

    X_train_observed_a = pd.read_parquet('../A/X_train_observed.parquet')
    X_train_observed_b = pd.read_parquet('../B/X_train_observed.parquet')
    X_train_observed_c = pd.read_parquet('../C/X_train_observed.parquet')

    X_test_estimated_a = pd.read_parquet('../A/X_test_estimated.parquet')
    X_test_estimated_b = pd.read_parquet('../B/X_test_estimated.parquet')
    X_test_estimated_c = pd.read_parquet('../C/X_test_estimated.parquet')

    return train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c

def build_corr_matrix(data_frames, figsize=(20,20), annot=True):
    plt.figure(figsize=figsize)
    sns.heatmap(data_frames.corr(method='pearson'), annot=annot, cmap=plt.cm.Reds)


# Source: https://sites.google.com/view/aide-python/statistiques/corr%C3%A9lations-et-corr%C3%A9logramme-en-python

def corrigraph(x,pval=0.05) :
    # Créer une matrice d'adjacence de type matrice de corrélation
    mycor = x.corr() # Matrice de corrélation non filtrée par les p-values
    # Nettoyage en fonction de la p-value
    titres_retenus = mycor.columns.values
    for i in range(len(titres_retenus)-1) :
        for j in range((i+1),len(titres_retenus)) :          
            titre = titres_retenus[i]
            #print("titre",titre)
            sous_titre = titres_retenus[j]
            #print("sous_titre",sous_titre)
            temp = x[[titre,sous_titre]].dropna()
            pvalue = scipy.stats.pearsonr(temp[titre],temp[sous_titre])[1]
            #print("pvalue",pvalue)
            if pvalue > pval :
                mycor.iloc[i,j] = 0
                mycor.iloc[j,i] = 0         
    mycor2 = np.array(mycor) # Conversion de la matrice pandas en matrice array   
    labels = list(mycor.columns.values) # Conversion au format list des noms de variables
    key_list = list( range(len(labels)))
    dic = dict(zip(key_list, labels))

    # Définir les couleurs en fonction des valeurs de corrélation

    # (>0 : rouge, <0 : bleu)

    G = nx.from_numpy_array(mycor2)
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    # Coloration conditionnelle
    weights  = list(weights)
    for i in range(len(weights)) :
        if weights[i] < 0 :
            weights[i] = "red"
        else :
            weights[i] = "blue"

    weights = tuple(weights)
    # Tracer le réseau

    # Nommer les labels
    G = nx.relabel_nodes(G, dic)
    position = nx.spring_layout(G)
    plt.figure(figsize=(6,6))
    nx.draw(G,position,with_labels=True,font_size=8, alpha=0.8,node_color="#FFFFD3",edge_color=weights)
    #nx.draw_networkx_edges(G, position, alpha=0.3,edgelist=edges,) 
    plt.show()
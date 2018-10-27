import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from keras.layers import RepeatVector, Dense, Activation, Lambda, Input, Flatten
from keras.layers import pooling
from keras.layers import convolutional
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import SVD, KNNWithZScore, KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise import dump
import matplotlib.pyplot as plt
import matplotlib

predictions, algo = dump.load("librarydata/model.md")
preditcions_knn, algo_knn = dump.load("librarydata/model_knn.md")

pu = algo.pu
qi = algo.qi
bu = algo.bu
bi = algo.bi

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0
    
    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(u, v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(np.square(u)))
    
    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(np.square(v)))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / (norm_u * norm_v)
    
    return cosine_similarity
	
	
def equalize(u,v, bias_axis):
    """
    Debias gender specific words by following the equalize method described in the figure above.
    
    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors
    
    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    
    # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
    e_w1, e_w2 = u, v
     
    # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
    mu = (e_w1 + e_w2) / 2

#     Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
    mu_B = np.dot(mu, bias_axis) / np.linalg.norm(bias_axis) + np.linalg.norm(bias_axis) * bias_axis
    mu_orth = mu - mu_B

    # Step 4: Set e1_orth and e2_orth to be equal to mu_orth (≈2 lines)
    e1_orth = mu_orth
    e2_orth = mu_orth
        
    # Step 5: Adjust the Bias part of u1 and u2 using the formulas given in the figure above (≈2 lines)
    e_w1B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * ((e_w1 - e1_orth) - mu_B) / np.abs(((e_w1 - e1_orth) - mu_B))
    e_w2B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * ((e_w2 - e2_orth) - mu_B) / np.abs(((e_w2 - e2_orth) - mu_B))
    
    # Step 6: Debias by equalizing u1 and u2 to the sum of their projections (≈2 lines)
    e1 = e_w1B + e1_orth
    e2 = e_w2B + e2_orth

    mu_B = np.dot(mu, bias_axis) / np.square(np.linalg.norm(bias_axis)) * bias_axis
    mu_orth = mu - mu_B

    e_w1B = np.dot(e_w1, bias_axis) / np.square(np.linalg.norm(bias_axis)) * bias_axis
    e_w2B = np.dot(e_w2, bias_axis) / np.square(np.linalg.norm(bias_axis)) * bias_axis
        
    corrected_e_w1B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * (e_w1B - mu_B) / np.linalg.norm(e_w1 - mu_orth - mu_B)
    corrected_e_w2B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * (e_w2B - mu_B) / np.linalg.norm(e_w2 - mu_orth - mu_B)

    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
    
    return e1, e2
	

	
def relation_avg(n, k):
    """
        第n位用户的k个邻居
        Return:
                基于距离的k个邻居
    """
    global algo_knn,pu
    if algo_knn is None:
        _, algo_knn = dump.load("librarydata/model_knn.md")
    neighbor = algo_knn.get_neighbors(n, k=2*k)
    neighbor_prev = algo_knn.get_neighbors(n, k=k)
    print(neighbor)
    print(neighbor_prev)
    if pu is None:
        pu = algo.pu
        qi = algo.qi
        bu = algo.bu
        bi = algo.bi
    g = [(0,100) for x in range(k)]
    sum_uv = 0
    print(g)
    for i in neighbor:
        uv = pu[n] - pu[i]
        sum_uv += uv
    avg_uv = sum_uv / (2*k)
    for i in neighbor:
        e1, e2 = equalize(pu[n], pu[i], avg_uv)
        relation_distance = np.abs(cosine_similarity(e1, avg_uv))
#         print(relation_distance)
        distance_min = max(g, key=lambda x:x[1])
        if relation_distance < distance_min[1]:
            min_index = g.index(distance_min)
#             print(min_index)
            g[min_index] = (i, relation_distance)
#         print(g)   
#     print(avg_uv)
    print (g)
    return [x[0] for x in g]
	
	

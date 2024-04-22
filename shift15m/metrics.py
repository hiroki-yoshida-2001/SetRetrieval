import tensorflow as tf
import matplotlib.pylab as plt
import os
import numpy as np
import pdb
import torch

def f1_bert_score(simMaps, score):
    """アイテム集合と予測ベクトルとのbertスコアを評価する関数"""
    """対角成分のスコアが上位10%にあれば1 そうでなければ0"""
    threk = int(len(score)*0.1)
    
    accuracy = np.zeros((len(score), 1))
    for batch_ind in range(len(score)):
        f1_score = score[batch_ind]
        sort_score, sort_index = tf.math.top_k(f1_score[:,0], k=threk)
        if batch_ind in sort_index:
            accuracy[batch_ind] += 1

    return accuracy
    #return tf.linalg.trace(score)

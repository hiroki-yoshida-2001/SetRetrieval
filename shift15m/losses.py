import tensorflow as tf
import matplotlib.pylab as plt
import os
import numpy as np
import pdb
import math

def F1_bert_hinge_loss(simMaps:tf.Tensor,scores:tf.Tensor)->tf.Tensor:
    #pdb.set_trace()
    # class_labels = tf.transpose(class_labels,[1,0])[0]
    scorepos = tf.argmax(scores)
    hingelosssum= []
    #Itemsize = set_labels.shape[1] 
    slack_variable = 0.2
    for batch_ind in range(len(simMaps)): 
        
        f1_score = scores[batch_ind]

        positive_score = tf.gather(f1_score, batch_ind)
        # n番目以外のインデックスの要素を取り出す
        negative_score = tf.gather(scores, tf.where(tf.not_equal(tf.range(tf.shape(f1_score)[0]), batch_ind)))

        hingeloss = tf.maximum(negative_score - positive_score + slack_variable , 0.0)
        hingelosssum.append(tf.reduce_mean(hingeloss))
        simMap = simMaps[batch_ind][batch_ind]
        sort_score, sort_indices = tf.math.top_k(simMap)
        positive_loss = 0
        '''
        if sort_indices[0] == sort_indices[1] == sort_indices[2]:
            positive_loss += (abs(sort_score[0]-sort_score[1])+abs(sort_score[1]-sort_score[2]))
        elif sort_indices[0] == sort_indices[2]:
            positive_loss += abs(sort_score[0]-sort_score[1])
        elif sort_indices[2] == sort_indices[1]:
            positive_loss += abs(sort_score[2]-sort_score[1])
        elif sort_indices[0] == sort_indices[1]:
            positive_loss += abs(sort_score[0]-sort_score[1])
        '''

    Loss = sum(hingelosssum)/len(hingelosssum) + 2.0*(positive_loss/len(simMaps)) 
    
    return Loss

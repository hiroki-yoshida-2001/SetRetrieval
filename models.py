import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import pdb
import sys
import math
import os
import pickle
from sklearn.decomposition import SparseCoder
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning
from SMscore_model import SetMatchingModel

warnings.filterwarnings("ignore", category=ConvergenceWarning)
#----------------------------
# normalization
class layer_normalization(tf.keras.layers.Layer):
    def __init__(self, size_d, epsilon=1e-3, is_set_norm=False, is_cross_norm=False):
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.is_cross_norm = is_cross_norm
        self.is_set_norm = is_set_norm

    def call(self, x, x_size):
        smallV = 1e-8
        # x : (nSet, nItemMax, d), x_size : (nSet, )
        if self.is_set_norm:
            if self.is_cross_norm:
                x = tf.concat([tf.transpose(x,[1,0,2,3]),x], axis=2)
                x_size=tf.expand_dims(x_size,-1)
                x_size_tile=x_size+tf.transpose(x_size)
            else:        
                shape = tf.shape(x)
                # x_size_tile = tf.tile(tf.expand_dims(x_size,1),[shape[1]])
            # change shape        
            shape = tf.shape(x)
            # x_reshape: (nSet, nItemMax * d)
            x_reshape = tf.reshape(x,[shape[0],-1])

            # zero-padding mask
            # mask : (nSet, nItemMax * d)
            mask = tf.reshape(tf.tile(tf.cast(tf.reduce_sum(x,axis=-1,keepdims=1)!=0,float),[1,1,shape[-1]]),[shape[0],-1])
            # mask = tf.cast(tf.not_equal(x_reshape,0),float)  
            # mean and std of set
            # mean_set : (nSet, )
            mean_set = tf.reduce_sum(x_reshape,-1)/(x_size*tf.cast(shape[-1],float))
            # diff : (nSet, nItemMax * d)
            diff = x_reshape-tf.tile(tf.expand_dims(mean_set,-1),[1,shape[1]*shape[2]])
            # std_set: (nSet, )
            std_set = tf.sqrt(tf.reduce_sum(tf.square(diff)*mask,-1)/(x_size*tf.cast(shape[-1],float)))
        
            # output
            # output : (nSet, nItemMax * d) => (nSet, nItemMax, d)
            output = diff/tf.tile(tf.expand_dims(std_set + smallV,-1),[1,shape[1]*shape[2]])*mask
            output = tf.reshape(output,[shape[0],shape[1],shape[2]])

            if self.is_cross_norm:
                output = tf.split(output,2,axis=2)[0]
        else:
            shape = tf.shape(x)

            # mean and std of items
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            std = tf.math.reduce_std(x, axis=-1, keepdims=True)
            norm = tf.divide((x - mean), std + self.epsilon)
            
            # zero-padding mask
            mask = tf.tile(tf.cast(tf.reduce_sum(x,axis=-1,keepdims=1)!=0,float),[1,1,1,shape[-1]])

            # output
            output = tf.where(mask==1, norm, tf.zeros_like(x))

        return output
#----------------------------
# multi-head CS function to make cros-set matching score map
class cross_set_score(tf.keras.layers.Layer):
    def __init__(self, head_size=20, num_heads=2):
        super(cross_set_score, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        #self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,kernel_constraint=tf.keras.constraints.NonNeg(),use_bias=False)
        self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,use_bias=False,name='CS_cnn')
        self.linear2 = tf.keras.layers.Dense(1,use_bias=False, name='CS_cnn')

    def call(self, x, nItem):
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))
        # compute inner products between all pairs of items with cross-set feature (cseft)
        # Between set #1 and set #2, cseft x[0,1] and x[1,0] are extracted to compute inner product when nItemMax=2
        # More generally, between set #i and set #j, cseft x[i,j] and x[j,i] are extracted.
        # Outputing (nSet_y, nSet_x, num_heads)-score map 
        
        # if input x is not tuple, existing methods are done.
        if not type(x) is tuple: # x :(nSet_x, nSet_y, nItemMax, dim)
            nSet_x = tf.shape(x)[0]
            nSet_y = tf.shape(x)[1]
            nItemMax = tf.shape(x)[2]

            # linear transofrmation from (nSet_x, nSet_y, nItemMax, Xdim) to (nSet_x, nSet_y, nItemMax, head_size*num_heads)
            x = self.linear(x)

            # reshape (nSet_x, nSet_y, nItemMax, head_size*num_heads) to (nSet_x, nSet_y, nItemMax, num_heads, head_size)
            # transpose (nSet_x, nSet_y, nItemMax, num_heads, head_size) to (nSet_x, nSet_y, num_heads, nItemMax, head_size)
            x = tf.transpose(tf.reshape(x,[nSet_x, nSet_y, nItemMax, self.num_heads, self.head_size]),[0,1,3,2,4])

            scores = tf.stack(
            [[
                tf.reduce_sum(tf.reduce_sum(
                tf.keras.layers.LeakyReLU()(tf.matmul(x[j,i],tf.transpose(x[i,j],[0,2,1]))/sqrt_head_size)
                ,axis=1),axis=1)/nItem[i]/nItem[j]
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
        else:
            x, y = x # x, y : (nSet_x(y), nItemMax, dim)
            nSet_x = tf.shape(x)[0]
            nSet_y = tf.shape(y)[0]
            nItemMax_x = tf.shape(x)[1]
            nItemMax_y = tf.shape(y)[1]

            # linear transofrmation from (nSet_x, nSet_y, nItemMax, Xdim) to (nSet_x, nSet_y, nItemMax, head_size*num_heads)
            x = self.linear(x)
            y = self.linear(y)
            # reshape (nSet_x (nSet_y), nItemMax, head_size*num_heads) to (nSet_x (nSet_y), nItemMax, num_heads, head_size)
            # transpose (nSet_x (nSet_y), nItemMax, num_heads, head_size) to (nSet_x (nSet_y), num_heads, nItemMax, head_size)
            x = tf.transpose(tf.reshape(x,[nSet_x, nItemMax_x, self.num_heads, self.head_size]),[0,2,1,3])
            y = tf.transpose(tf.reshape(y,[nSet_y, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])
            '''
            scores = tf.stack(
            [[
                tf.reduce_sum(tf.reduce_sum(
                tf.keras.layers.LeakyReLU()(tf.matmul(y[j],tf.transpose(x[i],[0,2,1]))/sqrt_head_size)
                , axis=1), axis=1)/nItem[i]/tf.cast(nItemMax_y, tf.float32)
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
            '''
            x_expand = tf.expand_dims(x, 1) # (nSet_x, 1, num_heads, nItemMax, head_size)
            y_expand = tf.expand_dims(y, 0) # (1, nSet_y, num_heads, nItemMax, head_size)
            scores = tf.keras.layers.LeakyReLU()(tf.einsum('aijkl,ibjml->abjkm', x_expand, y_expand)) / sqrt_head_size # (nSet_y, nSet_x, num_heads, nItemMax_x, nItemMax_y)
            scores = tf.reduce_sum(tf.reduce_sum(scores, axis=3), axis=3) / tf.reshape(nItem, (1, -1, 1)) / tf.cast(nItemMax_y, tf.float32) # (nSet_y, nSet_x, num_heads)

             
        # linearly combine multi-head score maps (nSet_y, nSet_x, num_heads) to (nSet_y, nSet_x, 1)
        scores = self.linear2(scores)

        return scores
#----------------------------


#----------------------------
# self- and cross-set attention
class set_attention(tf.keras.layers.Layer):
    def __init__(self, head_size=20, num_heads=2, activation="softmax", self_attention=False):
        super(set_attention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads        
        self.activation = activation
        self.self_attention = self_attention
        self.pivot_cross = False
        self.rep_vec_num = 1

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        # self.linearX = tf.keras.layers.Dense(units=self.head_size, use_bias=False, name='set_attention')
        # self.linearY = tf.keras.layers.Dense(units=self.head_size, use_bias=False, name='set_attention')
        self.linearQ = tf.keras.layers.Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_attention')
        self.linearK = tf.keras.layers.Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_attention')
        self.linearV = tf.keras.layers.Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_attention')
        self.linearH = tf.keras.layers.Dense(units=self.head_size, use_bias=False, name='set_attention')

    def call(self, x, y):
        # number of sets
        nSet_x = tf.shape(x)[0]
        nSet_y = tf.shape(y)[0]
        nItemMax_x = tf.shape(x)[1]
        nItemMax_y = tf.shape(y)[1]
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))

        # input (nSet, nSet, nItemMax, dim)
        # linear transofrmation (nSet, nSet, nItemMax, head_size*num_heads)
        y_K = self.linearK(y)   # Key
        y_V = self.linearV(y)   # Value
        x = self.linearQ(x)     # Query

        if self.pivot_cross: # pivot-cross
            y_K = tf.concat([y_K, x],axis=1)   # Key
            y_V = tf.concat([y_V, x],axis=1)   # Value            
            nItemMax_y += nItemMax_x

        # reshape (nSet, nItemMax, num_heads*head_size) to (nSet, nItemMax, num_heads, head_size)
        # transpose (nSet, nItemMax, num_heads, head_size) to (nSet, num_heads, nItemMax, head_size)
        x = tf.transpose(tf.reshape(x,[-1, nItemMax_x, self.num_heads, self.head_size]),[0,2,1,3])
        y_K = tf.transpose(tf.reshape(y_K,[-1, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])
        y_V = tf.transpose(tf.reshape(y_V,[-1, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])

        # inner products between all pairs of items, outputing (nSet, num_heads, nItemMax_x, nItemMax_y)-score map    
        xy_K = tf.matmul(x,tf.transpose(y_K,[0,1,3,2]))/sqrt_head_size

        def masked_softmax(x):
            # 0 value is treated as mask
            mask = tf.not_equal(x,0)
            x_exp = tf.where(mask,tf.exp(x-tf.reduce_max(x,axis=-1,keepdims=1)),tf.zeros_like(x))
            softmax = x_exp/(tf.reduce_sum(x_exp,axis=-1,keepdims=1) + 1e-10)

            return softmax

        # normalized by softmax
        attention_weight = masked_softmax(xy_K)
        # computing weighted y_V, outputing (nSet, num_heads, nItemMax_x, head_size)
        weighted_y_Vs = tf.matmul(attention_weight, y_V)

        # reshape (nSet, num_heads, nItemMax_x, head_size) to (nSet, nItemMax_x, head_size*num_heads)
        weighted_y_Vs = tf.reshape(tf.transpose(weighted_y_Vs,[0,2,1,3]),[-1, nItemMax_x, self.num_heads*self.head_size])
        
        # combine multi-head to (nSet, nItemMax_x, head_size)
        output = self.linearH(weighted_y_Vs)

        return output
#----------------------------
# Linear Projection (SHIFT15M => head_size) 
class MLP(tf.keras.Model):
    def __init__(self, baseChn=1024, category_class_num=41,dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(baseChn, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.fc2 = tf.keras.layers.Dense(baseChn//2, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.fc3 = tf.keras.layers.Dense(baseChn//4, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn')
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.fc4 = tf.keras.layers.Dense(category_class_num, activation='softmax', use_bias=False, name='setmatching_cnn')

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        output = self.fc4(x)

        return x, output
    
    def train_step(self, data):
        
        x, y, class_weights = data

        sample_weights = tf.gather(class_weights, tf.cast(y, tf.int32))

        with tf.GradientTape() as tape:
            _, y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
     
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, trainable_vars)
            if grad is not None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):

        x, y = data
    
        # predict
        _, y_pred = self(x, training=False)
        
        # loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}
#----------------------------

#----------------------------
# CNN
class CNN(tf.keras.Model):
    def __init__(self, baseChn=32, cnn_class_num=2, num_conv_layers=3, max_channel_ratio=2):
        super(CNN, self).__init__()
        self.baseChn = baseChn
        self.num_conv_layers = num_conv_layers

        self.convs = [tf.keras.layers.Conv2D(filters=baseChn*np.min([i+1,max_channel_ratio]), strides=(2,2), padding='same', kernel_size=(3,3), activation='relu', use_bias=False, name='class') for i in range(num_conv_layers)]
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()

        self.fc_cnn_final1 = tf.keras.layers.Dense(baseChn, activation='relu', name='class')
        self.fc_cnn_final2 = tf.keras.layers.Dense(cnn_class_num, activation='softmax', name='class')

    def call(self, x):
        x, x_size = x

        # reshape (nSet, nItemMax, H, W, C) to (nSet*nItemMax, H, W, C)
        shape = tf.shape(x)
        nSet = shape[0]
        nItemMax = shape[1]
        x = tf.reshape(x,[-1,shape[2],shape[3],shape[4]])
        debug = {}

        # CNN
        for i in range(self.num_conv_layers):
            x = self.convs[i](x)
        x = self.globalpool(x)
        
        # classificaiton of set
        output = self.fc_cnn_final1(tf.reshape(x,[nSet,-1]))
        output = self.fc_cnn_final2(output)

        return x, output

    # train step
    def train_step(self,data):
        x, y_true = data
        x, x_size = x

        with tf.GradientTape() as tape:
            # predict
            _, y_pred = self((x, x_size), training=True)

            # loss
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
     
        # train using gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, trainable_vars)
            if grad is not None)

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):
        x, y_true = data
        x, x_size = x

        # predict
        _, y_pred = self((x, x_size), training=False)
        
        # loss
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}
#----------------------------

# ---------------山園追加部分------------------
# class SetMatchingModel(tf.keras.Model)
    # ここに集合マッチングモデルを
# --------------------------------------------
class CustomMetric(tf.keras.metrics.Metric):
    def __init__(self, name='custom_metric', **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')  # 値の合計
        self.count = self.add_weight(name='count', initializer='zeros')  # サンプル数

    def update_state(self, y, metric_value, sample_weight=None):
        """
        metric_value: compute_custom_metricで計算されたメトリクスの値
        sample_weight: 任意の重み（指定されない場合はNone）
        """
        # # metric_valueを累積する
        # if not isinstance(metric_value, tf.Tensor):
        #     metric_value = tf.cast(tf.constant(metric_value), tf.float32)
        # else:
        #     metric_value = tf.cast(metric_value, tf.float32)
        batch_size = tf.shape(metric_value)[0]
        mean_metric_value = tf.reduce_mean(metric_value)
        # バッチごとの平均にバッチサイズを掛けて、加重平均を累積
        self.total.assign_add(tf.cast(mean_metric_value * batch_size.numpy(), tf.float32))

        # 全サンプル数を累積
        self.count.assign_add(tf.cast(batch_size, tf.float32))
        # self.total.assign_add(metric_value)
        
        # # サンプル数をカウント
        # self.count.assign_add(tf.cast(tf.size(metric_value), tf.float32))

    def result(self):
        # 合計値をサンプル数で割って平均を返す
        return self.total / self.count

    def reset_state(self):
        # 状態をリセット
        self.total.assign(0)
        self.count.assign(0)

#----------------------------
# set matching network
class SMN(tf.keras.Model):
    def __init__(self, isCNN=True, is_set_norm=False, is_TrainableMLP=True, num_layers=1, num_heads=2, calc_set_sim='BERTscore', baseChn=32, baseMlp = 512, rep_vec_num=1, seed_init = 0, max_channel_ratio=2, is_Cvec_linear=False, use_all_pred=False, is_category_emb=False, c1_label=True, cos_sim_loss = 'CE', style_loss = 'item_style'):
        super(SMN, self).__init__()
        self.isCNN = isCNN
        self.num_layers = num_layers
        self.calc_set_sim = calc_set_sim
        self.rep_vec_num = rep_vec_num
        self.seed_init = seed_init
        self.baseChn = baseChn
        self.isTrainableMLP = is_TrainableMLP
        self.baseMlpChn = baseMlp
        self.is_Cvec_linear = is_Cvec_linear
        self.use_all_pred = use_all_pred
        self.is_category_emb = is_category_emb
        self.is_c1label = c1_label
        self.cos_sim_method = cos_sim_loss
        self.style_method = style_loss
        
        if self.seed_init != 0:
            self.dim_shift15 = len(self.seed_init[0])
        
        #---------------------
        # cnn
        self.CNN = []
        self.fc_cnn_proj = tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn')
        #---------------------
        # projection layer for pred
        self.fc_pred_proj = tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn') # nameにcnn
        #---------------------
        # encoder for query X
        self.set_emb = self.add_weight(name='set_emb',shape=(1,self.rep_vec_num,baseChn*max_channel_ratio),trainable=True)
        self.self_attentionsX = [set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads, self_attention=True) for i in range(num_layers)]
        self.layer_norms_enc1X = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.layer_norms_enc2X = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.fcs_encX = [tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching') for i in range(num_layers)]        
        #---------------------

        #---------------------
        # decoder
        self.cross_attentions = [set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads) for i in range(num_layers)]
        self.layer_norms_dec1 = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.layer_norms_dec2 = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.layer_norms_decq = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.layer_norms_deck = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.fcs_dec = [tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching') for i in range(num_layers)]
        #---------------------
     
        #---------------------
        # head network
        self.cross_set_score = cross_set_score(head_size=baseChn*max_channel_ratio, num_heads=num_heads)
        self.pma = set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads)  # poolingMA
        self.fc_final1 = tf.keras.layers.Dense(baseChn, name='setmatching')
        self.fc_final2 = tf.keras.layers.Dense(1, activation='sigmoid', name='setmatching')
        self.fc_proj = tf.keras.layers.Dense(1, use_bias=False, name='projection')  # linear projection
        #---------------------
        self.c2toc1 = [[0,1,2,3,4],[5,6,7,8,9,10,11,12],[13,14,15,16,17],[18,19,20,21,22],[23,24,25,26,27,28,29],[30,31,32,33,34,35,36],[37,38,39,40]]
        self.label_slice = False # if slice emb vector before input or loss calculation T : before slicing F : After slicing
        self.cluster_moveable = False
        self.key_cluster = True
        #---------------------


        # ---------------山園追加部分------------------
        # マッチングモデルの初期化定義 self.Matching = []とか
        self.SetMatchingModel = SetMatchingModel()
        
        # --------------------------------------------
        # seed_vec initialization with cluster vectors
        if self.seed_init == 0:
            self.set_emb = self.add_weight(name='set_emb',shape=(1, self.rep_vec_num,baseChn*max_channel_ratio),trainable=True)
        else:
            if self.is_c1label:
                if self.cluster_moveable:
                    self.cluster_emb = [self.add_weight(name='set_emb',shape=(self.dim_shift15,),initializer=self.custom_initializer(self.seed_init[i]),trainable=True) for i in range(len(self.seed_init))]
                    self.set_emb = []
                    for i in range(len(self.c2toc1)):
                        added_vector = self.add_weight(name='set_emb',shape=(self.dim_shift15,),initializer=self.custom_initializer(self.cluster_emb[self.c2toc1[i][0] : self.c2toc1[i][-1]], c2toc1=True),trainable=True)
                        self.set_emb.append(added_vector)
                    self.cluster = [self.add_weight(name='set_emb',shape=(self.dim_shift15,),initializer=self.custom_initializer(self.seed_init[i]),trainable=True) for i in range(len(self.seed_init))]
                else:
                    self.set_emb = []
                    for i in range(len(self.c2toc1)):
                        added_vector = self.add_weight(name='set_emb',shape=(self.dim_shift15,),initializer=self.custom_initializer(self.seed_init[self.c2toc1[i][0] : self.c2toc1[i][-1]], c2toc1=True),trainable=True)
                        self.set_emb.append(added_vector)
                    self.cluster = [self.add_weight(name='set_emb',shape=(self.dim_shift15,),initializer=self.custom_initializer(self.seed_init[i]),trainable=True) for i in range(len(self.seed_init))]
            else:
                # 4096 => 64次元への写像処理が必要
                self.set_emb = [self.add_weight(name='set_emb',shape=(self.dim_shift15,),initializer=self.custom_initializer(self.seed_init[i]),trainable=True) for i in range(len(self.seed_init))]
        #---------------------
        # MLP models
        self.MLP = []

        # category embedded vector
        # self.category_emb = [self.add_weight(name='category_emb',shape=(self.dim_shift15,),initializer=self.custom_initializer(self.seed_init[i], category_emb=True),trainable=True) for i in range(len(self.seed_init))]

        
    def custom_initializer(self, initial_values, category_emb=False, c2toc1=False):
        def initializer(shape, dtype=None):
            # 次元ごとに異なる値を持つTensorを作成
            if category_emb:
                values = [1/100 * initial_values[i] for i in range(shape[0])]
            else:
                if not c2toc1:
                    values = [initial_values[i] for i in range(shape[0])]
                else:
                    initial_values_np = np.array(initial_values)
                    # 各次元ごとに平均を求める
                    if self.cluster_moveable:
                        values = np.sum(initial_values_np) / initial_values_np.shape
                    else:
                        values = np.mean(initial_values_np, axis=0)
            return tf.constant(values, dtype=dtype)
        return initializer
    
    def call(self, x):
        if not self.use_all_pred:
            x, x_size, c_label, c1_label, y_pred_size = x
        else:
            x, x_size = x
        debug = {}
        shape = tf.shape(x)
        nSet = shape[0]
        nItemMax = shape[1]

        # category embeddeing
        if self.is_category_emb and not self.use_all_pred:
            x  += tf.gather(tf.stack(self.category_emb), c_label)
        # CNN
        if self.isCNN:
            x, predCNN = self.CNN((x,x_size),training=False)
        else:
            if self.isTrainableMLP:
                x, _ = self.MLP(x,training=False)

            else:
                x = self.fc_cnn_proj(x) # input: (nSet, nItemMax, D=4096) output:(nSet, nItemMax, D=64(baseChn*max_channel_ratio))
            predCNN = []
        
        
        debug['x_encoder_layer_0'] = x

        x_2enc = x

        #---------------------
        # encoder (self-attention)
        # for query x
        for i in range(self.num_layers):

            z = self.layer_norms_enc1X[i](x,x_size)

            # input: (nSet, nItemMax, D), output:(nSet, nItemMax, D)
            z = self.self_attentionsX[i](z,z)
            x += z

            z = self.layer_norms_enc2X[i](x,x_size)
            z = self.fcs_encX[i](z)
            x += z

            debug[f'x_encoder_layer_{i+1}'] = x
        x_enc = x
        #---------------------

        #---------------------
        if self.isTrainableMLP:
            if self.seed_init == 0:
                y_pred = tf.tile(self.set_emb, [nSet, 1,1])
            else:
                if not self.use_all_pred:
                    if self.is_c1label: # 7 category ver
                        if self.label_slice:
                            y_pred = tf.gather(tf.stack(self.set_emb), c1_label)
                            cluster = tf.tile(tf.expand_dims(tf.stack(self.cluster), axis=0),[nSet,1,1])
                        else:
                            y_pred = tf.tile(tf.expand_dims(tf.stack(self.set_emb), axis=0),[nSet,1,1])
                            cluster = tf.tile(tf.expand_dims(tf.stack(self.cluster), axis=0),[nSet,1,1])
                    else: # 41category ver
                        if self.label_slice:
                            y_pred = tf.gather(tf.stack(self.set_emb), c_label)
                        else:
                            y_pred = tf.tile(tf.expand_dims(tf.stack(self.set_emb), axis=0),[nSet,1,1])
                else:
                    y_pred = tf.tile(tf.expand_dims(tf.stack(self.set_emb), axis=0),[nSet,1,1])
                y_pred, _ = self.MLP(y_pred, training=False)
                
                # cluster K, Q
                cluster, _ = self.MLP(cluster, training=False)
                
        else:
            # add_embedding
            if self.seed_init == 0:
                y_pred = tf.tile(self.set_emb, [nSet,1,1]) # (nSet, nItemMax, D)
            else:
                y_pred = tf.tile(tf.expand_dims(tf.stack(self.set_emb), axis=0),[nSet,1,1])
                if self.is_Cvec_linear:
                    y_pred = self.fc_pred_proj(y_pred) # y_pred = self.fc_cnn_proj(y_pred) # y_pred = self.fc_pred_proj(y_pred)
                else:
                    y_pred = self.fc_cnn_proj(y_pred)
        if self.use_all_pred:
            y_pred_size = tf.constant(np.full(nSet,self.rep_vec_num).astype(np.float32))
        elif not self.label_slice:
            y_pred_size = tf.constant(np.full(nSet,len(self.c2toc1)).astype(np.float32))
        
        #---------------------
        # decoder (cross-attention)
        debug[f'x_decoder_layer_0'] = x
        for i in range(self.num_layers):
    
            self.cross_attentions[i].pivot_cross = True

            query = self.layer_norms_decq[i](y_pred,y_pred_size)
            # key = self.layer_norms_deck[i](x,x_size)

            # cluster key, value
            if self.key_cluster:
                key = self.layer_norms_deck[i](tf.concat([x, cluster],axis=1), x_size+tf.constant(np.full(nSet,len(self.cluster)).astype(np.float32)))
            else:
                key = self.layer_norms_deck[i](x,x_size)
            # input: (nSet, nItemMax, D), output:(nSet, nItemMax, D)
            query = self.cross_attentions[i](query,key)
            y_pred += query
    
            query = self.layer_norms_dec2[i](y_pred,y_pred_size)
            

            query = self.fcs_dec[i](query)
            y_pred += query

            debug[f'x_decoder_layer_{i+1}'] = x
        x_dec = x

        return predCNN, y_pred, debug
    
    # compute cosine similarity between all pairs of items and BERTscore.
    def BERT_set_score(self, x, nItem,beta=0.2):
        
        # cos_sim : compute cosine similarity  between all pairs of items
        # -----------------------------------------------------
        # Outputing (nSet_y, nSet_x, nItemMax, nItemMax)
        # e.g, cos_sim[1][0] (nItemMax_y, nItemMax_x) means cosine similarity between y[1] (nItemMax, dim) and x[0] (nItemMax, dim)
        # -----------------------------------------------------

        # f1_scores : BERT_score with cos_sim
        # ------------------------------------------------------
        # Outputing (nSet_y, nSet_x, 1)
        # Using cos_sim , precision (y_neighbor) and recall (x_neighbor) are caluculated.
            # As to caluclating x_neighbor (recall), max score is extracted in row direction => tf.reduce_max(cos_sim[i], axis=2) : (nSet_x, nItemMax)
            # average each item score  => tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=2), axis=1, keepdims=True) / tf.expand_dims(nItem,axis=1) : (nSet_x, 1)
        
            # As to caluclating y_neighbor (precision), max score is extracted in column direction (nItemMax_y) -> tf.reduce_max(cos_sim[i], axis=1) : (nSet_x, nItemMax)
            # average each item score =>  tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=1), axis=1, keepdims=True) / tf.expand_dims(y_item,axis=1) : (nSet_x, 1)
            # for precision caluculating, -inf masking is processed before searching neighbor score.
            # if cos_sim[i] has 0 value , we regard the value as a similarity between y[i] and zero padding item in x, replacing -inf in order for not choosing . 
            # tf.where(tf.not_equal(cos_sim[i], 0), cos_sim[i], tf.fill(cos_sim[i].shape, float('-inf')))
        
        # f1_scores = 2 * (y_neighbor * x_neighbor) / (y_neighbor + x_neighbor)
        # e.g, f1_scores[0,1] (,1) means BERT_score (set similarity) between y[0] and x[1]
        # ------------------------------------------------------

        if not type(x) is tuple: # x :(nSet_x, nSet_y, nItemMax, dim)
            nSet_x = tf.shape(x)[0]
            nSet_y = tf.shape(x)[1]
            nItemMax = tf.shape(x)[2]

            cos_sim = tf.stack(
            [[                
                tf.matmul(tf.nn.l2_normalize(x[j,i], axis=-1),tf.transpose(tf.nn.l2_normalize(x[i,j], axis=-1),[0,2,1]))
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
            f1_scores = [
            
                2 * (
                    tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=1), axis=1, keepdims=True) / tf.expand_dims(nItem,axis=1) *
                    tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=2), axis=1, keepdims=True) / tf.expand_dims(nItem,axis=1)
                ) / (
                    tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=1), axis=1, keepdims=True) / tf.expand_dims(nItem,axis=1) +
                    tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=2), axis=1, keepdims=True) / tf.expand_dims(nItem,axis=1)
                )
                for i in range(len(cos_sim))
            ]
        else:
            x, y = x # x, y : (nSet_x(y), nItemMax, dim)
            nSet_x = tf.shape(x)[0]
            nSet_y = tf.shape(y)[0]
            nItemMax_y = tf.shape(y)[1]

            x_expand = tf.expand_dims(tf.nn.l2_normalize(x, axis=-1), 0) # (nSet_x, 1, nItemMax, head_size)
            y_expand = tf.expand_dims(tf.nn.l2_normalize(y, axis=-1), 1) # (1, nSet_y, nItemMax, head_size)
            cos_sim = tf.einsum('aijk,ibmk->abjm', y_expand, x_expand) # (nSet_y, nSet_x, nItemMax_x, nItemMax_y)
            if self.use_all_pred:
                f1_scores = [
                
                    2 * (
                        tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=1), axis=1, keepdims=True) / tf.expand_dims(nItem,axis=1) * # precision
                        tf.reduce_sum(tf.reduce_max(tf.where(tf.not_equal(cos_sim[i], 0), cos_sim[i], tf.fill(cos_sim[i].shape, float('-inf'))), axis=2), axis=1, keepdims=True) / tf.cast(nItemMax_y, tf.float32) # recall
                    ) / (
                        tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=1), axis=1, keepdims=True) / tf.expand_dims(nItem,axis=1) +
                        tf.reduce_sum(tf.reduce_max(tf.where(tf.not_equal(cos_sim[i], 0), cos_sim[i], tf.fill(cos_sim[i].shape, float('-inf'))), axis=2), axis=1, keepdims=True) / tf.cast(nItemMax_y, tf.float32)
                    )
                    for i in range(len(cos_sim))
                ]
            else:
                f1_scores = [tf.reduce_sum(tf.linalg.diag_part(cos_sim[i]), axis=-1, keepdims=True) / tf.expand_dims(nItem, axis=1)
                    for i in range(len(cos_sim))
                ]
  
        # ------------------------------
        f1_scores = tf.stack(f1_scores, axis=0)
        
        return f1_scores
    # calculate item wise cosine similarity
    def cos_similarity(self, x, PIFR=False):
        if not PIFR:
            x, y = x # x, y : (nSet_x(y), nItemMax, dim)
            x_expand = tf.expand_dims(tf.nn.l2_normalize(x, axis=-1), 0) # (nSet_x, 1, nItemMax, head_size)
            y_expand = tf.expand_dims(tf.nn.l2_normalize(y, axis=-1), 1) # (1, nSet_y, nItemMax, head_size)
            cos_sim = tf.einsum('aijk,ibmk->abjm', y_expand, x_expand) # (nSet_y, nSet_x, nItemMax_x, nItemMax_y)
        else:
            x, y = x # x, y : (nSet_x(y), nItemMax, dim)
            # x_expand = tf.expand_dims(tf.nn.l2_normalize(x, axis=-1), 0) # (nSet_x, 1, nItemMax, head_size)
            # y_expand = tf.expand_dims(tf.nn.l2_normalize(y, axis=-1), 1) # (1, nSet_y, nItemMax, head_size)
            # cos_sim = [tf.einsum('aijk,ibmk->abjm', tf.expand_dims(tf.nn.l2_normalize(y[i], axis=-1), 1), tf.expand_dims(tf.nn.l2_normalize(x[i], axis=-1), 0))for i in range(len(x))] # (nSet_y, nSet_x, nItemMax_x, nItemMax_y)
            cos_sim = tf.stack([tf.einsum('ik,jbk->jib', tf.nn.l2_normalize(y[i], axis=-1), tf.nn.l2_normalize(x[i], axis=-1)) for i in range(len(x))])
            
        return cos_sim
    
    def gram_matrix(self, input_tensor):
        # result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

        result = tf.einsum('bnd,bne->bnde', input_tensor, input_tensor) # pixel loss
        # result = tf.einsum('bnd,bnd->bn', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)
    
    def style_content_loss(self, pred, ans, pred_size):
        # style_outputs = outputs['style']
        # content_outputs = outputs['content']
        # style pixel loss ver.
        pred_gram = self.gram_matrix(pred)
        if self.style_method == 'item_style':
            ans_gram = self.gram_matrix(ans)
        elif self.style_method == 'DFA_style':
            ans_gram = tf.stack([self.gram_matrix(ans[i]) for i in range(len(pred))]) # PIFR ver
            ans_gram = tf.stack([ans_gram[i,i,:,:,:] for i in range(len(pred))])
        
        gram_shape = pred_gram.shape
        pred_gram = tf.reshape(pred_gram, [gram_shape[0], gram_shape[1], gram_shape[2]*gram_shape[3]])
        ans_gram = tf.reshape(ans_gram, [gram_shape[0], gram_shape[1], gram_shape[2]*gram_shape[3]])
        
        item_style_loss = tf.reduce_sum(tf.reduce_mean(tf.square(pred_gram-ans_gram),axis=-1)) / pred_size
        
        # item_style_loss = tf.reduce_sum(tf.square(self.gram_matrix(pred)-self.gram_matrix(ans)),axis=-1) / pred_size
        style_loss = tf.reduce_mean(item_style_loss)

        return style_loss
    
    def SetMatchingScore_loss(self, g_Real, g_fake):
        Loss  = tf.math.log(1.0 + tf.exp(g_Real - g_fake))
        return tf.reduce_mean(Loss)
    # convert class labels to cross-set label（if the class-labels are same, 1, otherwise 0)
    def cross_set_label(self, y):
        # rows of table
        y_rows = tf.tile(tf.expand_dims(y,-1),[1,tf.shape(y)[0]])
        # cols of table       
        y_cols = tf.tile(tf.transpose(tf.expand_dims(y,-1)),[tf.shape(y)[0],1])

        # if the class-labels are same, 1, otherwise 0
        labels = tf.cast(y_rows == y_cols, float)            
        return labels

    def toBinaryLabel(self,y):
        dNum = tf.shape(y)[0]
        y = tf.map_fn(fn=lambda x:0 if tf.less(x,0.5) else 1, elems=tf.reshape(y,-1))

        return tf.reshape(y,[dNum,-1])

    def evaluate_inverse_ranks(self, array, target):
        valid_indices = np.where(~np.isnan(array))[0]
        valid_array = array.numpy()[valid_indices]
        sorted_indices = np.argsort(valid_array)[::-1]
        sorted_array = valid_array[sorted_indices]

        if math.isnan(target):
            ranks = -1
        else:
            rank = np.where(sorted_array == target)
            if len(rank) > 0:
                ranks = rank[0][0] + 1  # インデックスを1ベースに変換し、最も低いランクを追加
                ranks = ranks / len(sorted_array)
            else:
                ranks = -1  # 要素が見つからない場合は-1を追加

        return ranks

    def set_retrieval_rank(self, cos_sim, y_true, c_label):      
        # cos_sim, y_true, c_label = batch_data
        # 必要な計算をここで行います
        batch_size = c_label.shape[0]
        num_items = c_label.shape[1]

        # True indices (which batch corresponds to the true class for each sample)
        true_indices = tf.argmax(y_true, axis=1)  # shape: (batch_size,)

        # Initialize a list to accumulate the hinge loss for each batch
        Batch_ranks = []
        for batch_ind in range(batch_size):
            ranks = []
            for item_ind in range(num_items):
                true_class_label = c_label[true_indices[batch_ind]][item_ind]

                # Skip if the class label is 41
                if true_class_label != 41:
                    # Get indices where the class labels match the true class label
                    indices = tf.where(c_label == true_class_label)

                    # If there's only one match (i.e., the positive example itself)
                    if len(indices[:, 0]) == 1:
                        ranks.append(0)
                    else:
                        # Cosine similarity for the true positive
                        positive_score = cos_sim[batch_ind, :, item_ind, item_ind][true_indices[batch_ind]]
                        target_scores = tf.gather_nd(cos_sim[batch_ind, :,item_ind,:], indices)
                        
                        # Cosine similarities for the negative examples
                        # target_scores = tf.stack([
                        #     cos_sim[indices[i, 0], item_ind, indices[i, 1]]
                        #     for i in range(indices.shape[0])
                        # ])
                        ranks.append(self.evaluate_inverse_ranks(target_scores, positive_score))

                else:
                    ranks.append(-1)

            # Calculate the average loss for valid items
            filtered_values = [v for v in ranks if v != -1]
            if len(filtered_values) == 0:
                Batch_ranks.append(0)
            Batch_ranks.append(sum(filtered_values) / len(filtered_values))

        # # Calculate the average loss across all batches
        # Ranks = tf.reduce_mean(tf.stack(Batch_ranks))
        
        return tf.stack(Batch_ranks)
        # return tf.ones_like(c_label)
    def return_pred_label(self, cos_sim, c_label): 
        # cos_sim: shape (Batch, Batch, Items, 5)
        # c_label: shape (Batch, 5)
        # まず、(Batch, Items, Batch * 5) の形にreshapeする
        target_cos_sim = tf.transpose(cos_sim, [0,2,1,3])
        target_cos_sim = tf.reshape(target_cos_sim, [target_cos_sim.shape[0], target_cos_sim.shape[1], -1])  # (Batch, Items, Batch*5)
        
        # 最大のコサイン類似度のインデックスを取得 (Batch, Items)
        indices = tf.argmax(target_cos_sim, axis=-1)
        
        # (Batch, 5)の形にラベルをreshapeして全体でflattenする
        flattened_labels = tf.reshape(c_label, [-1])  # (Batch * 5)
        
        # 最大コサイン類似度に対応するラベルを取得 (Batch, Items)
        pred_labels = tf.gather(flattened_labels, indices)
        
        return pred_labels

    # # sparse representation in PIFR
    def compute_representation_weights(self, x, y, nItem, mode='l1', transform_algorithm='lasso_lars', transform_alpha=1):

        def compute_sparse_weight(X, Y, nItem_x, nItem_y, nItemMax, transform_algorithm, transform_alpha):
            coder = SparseCoder(dictionary=Y, transform_algorithm=transform_algorithm, transform_alpha=transform_alpha/nItem_y)
            weight = coder.transform(X)

            return weight

        def compute_collaborative_weight(X, Y, nItem_x, nItem_y, nItemMax, transform_alpha):
            nItem_x = np.min([nItem_x,nItemMax])
            nItem_y = np.min([nItem_y,nItemMax])

            X = X[:nItem_x]
            Y = Y[:nItem_y]

            term1 = np.matmul(X,np.transpose(Y))
            term2 = np.linalg.inv(np.matmul(Y,np.transpose(Y)) + transform_alpha/nItem_y * np.eye(nItem_y,nItem_y))
            weight = np.matmul(term1,term2)

            if nItem_y < nItemMax:
                weight = np.hstack([weight,np.zeros([nItem_x,nItemMax-nItem_y])])

            if nItem_x < nItemMax:
                weight = np.vstack([weight,np.zeros([nItemMax-nItem_x,nItemMax])])

            return weight


        try:
            x = x.numpy()
            y = y.numpy()
            nItem = nItem.numpy()
            nSet_x = x.shape[0]
            nSet_y = y.shape[0]
            nItemMax = x.shape[1]

            if mode == 'l1':
                # 求まるのはあくまでも係数
                weights = Parallel(n_jobs=-1)(delayed(compute_sparse_weight)(x[i], y[j], int(nItem[i]), int(nItem[j]), nItemMax, transform_algorithm, transform_alpha) for i in range(nSet_x) for j in range(nSet_y))
                #weights = [[compute_sparse_weight(x[i], y[j], int(nItem[i]), int(nItem[j]), nItemMax, transform_algorithm, transform_alpha) for i in range(nSet_x) for j in range(nSet_y)]]
            elif mode == 'l2':
                #weights = compute_collaborative_weight(x[0,1],x[1,0], int(nItem[0]), int(nItem[1]), nItemMax, transform_alpha)
                weights = Parallel(n_jobs=-1)(delayed(compute_collaborative_weight)(x[i,j], x[j,i], int(nItem[i]), int(nItem[j]),  nItemMax, transform_alpha) for i in range(nSet_x) for j in range(nSet_y))

            weights = np.stack(weights).reshape(nSet_x,nSet_y,nItemMax,nItemMax)

        except:
            pdb.set_trace()

        return tf.convert_to_tensor(weights,dtype=tf.float32)
    
    def PIFR_Loss(self, score, y_true, Loss_method='Hinge'):
        
        if Loss_method == 'Origin':      
            Mased_score = score * (2 * y_true - 1)
            result = tf.reduce_mean(Mased_score, axis=1)
        elif Loss_method == 'CE':
            # 交差エントロピー損失の計算 (Xをlogitsとして利用)
            bce = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            result = bce(y_true=y_true, y_pred=score)
        elif Loss_method == 'Hinge':
            '''
            hinge_losssum = []
            for batch_ind in range(len(y_true)): 
                target = score[batch_ind]

                positive_score = tf.boolean_mask(target, tf.equal(y_true[batch_ind], 1))
                negative_score = tf.boolean_mask(target, tf.equal(y_true[batch_ind], 0))

                # hingeloss : (nSet_y - 1, )
                hingeloss = tf.maximum(negative_score - positive_score + 0.2 , 0.0)
                hinge_losssum.append(tf.reduce_sum(hingeloss))
            result = tf.stack(hinge_losssum)
            '''
            # Yが1の部分をマスクとして使い、ポジティブスコアを取得
            positive_scores = tf.reduce_sum(score * y_true, axis=1)  # 各行のポジティブスコア (shape: [batch_size])

            # Yが0の部分をマスクして、ネガティブスコアを取得
            negative_scores = score * (1 - y_true)  # 各行でYが0のスコア (shape: [batch_size, num_classes])

            # ポジティブスコアをネガティブスコアにブロードキャストし、max(negative - positive + slack, 0.0) を計算
            hinge_loss = tf.maximum(negative_scores - tf.expand_dims(positive_scores, axis=1) + 0.2, 0.0)

            # 各行ごとのヒンジ損失を合計
            result = tf.reduce_sum(hinge_loss, axis=1)
        
        return tf.reduce_mean(result)
    def style_gram_checker(self, gallery):
        
        x_expand = tf.expand_dims(tf.nn.l2_normalize(gallery, axis=-1), 0) # (nSet_x, 1, nItemMax, head_size)
        y_expand = tf.expand_dims(tf.nn.l2_normalize(gallery, axis=-1), 1) # (1, nSet_y, nItemMax, head_size)
        cos_sim = tf.einsum('aijk,ibmk->abjm', y_expand, x_expand)
        pred_gram = self.gram_matrix(gallery)
        gram_shape = pred_gram.shape
        pred_gram = tf.reshape(pred_gram, [gram_shape[0], gram_shape[1], gram_shape[2]*gram_shape[3]])
        X_1 = tf.expand_dims(pred_gram, axis=0)
        X_2 = tf.expand_dims(pred_gram, axis=1)


        all_comb_dist = np.square(X_1.numpy()-X_2.numpy())
        MSE = np.mean(all_comb_dist, axis=-1)
        
        Top_5_index = []
        Top_5_cos = []
        
        for Batch_ind in range(len(MSE)):
            item_index = []
            cos_item_index = []
            for item_ind in range(MSE.shape[-1]):
                # masked_arr = np.where(MSE[Batch_ind][:,item_ind] == 0, np.inf, MSE[Batch_ind])
                flattened_indices = np.argsort(MSE[Batch_ind][:,item_ind], axis=None)
                flattened_cos = np.argsort(cos_sim[Batch_ind][:,item_ind,:], axis=None)[::-1]
                flattened_cos = np.stack(np.unravel_index(flattened_cos, cos_sim[0][:,0,:].shape),axis=1)
                top_5_indices = flattened_indices[:5]
                top_5_cos = flattened_cos[:5]
                # top_5_indices_2d = np.unravel_index(top_5_indices, MSE[Batch_ind][:,item_ind].shape)
                # top_5_indices_2d_array = np.stack(top_5_indices_2d, axis=1)
                item_index.append(top_5_indices)
                cos_item_index.append(top_5_cos)
            Top_5_index.append(item_index)
            Top_5_cos.append(cos_item_index)

        
        return MSE , np.stack(Top_5_index), np.stack(Top_5_cos)# (Batch, Batch, Nitem)
    # custom loss (Tensorflow does not support more than 3 types input, so write loss in this class...)
    def item_hinge_loss(self, cos_sim, y_true, c_label, pred, ans, pred_size):
        slack_variable = 0.2
        batch_size = c_label.shape[0]
        num_items = c_label.shape[1]
       
        # True indices (which batch corresponds to the true class for each sample)
        true_indices = tf.argmax(y_true, axis=1)  # shape: (batch_size,)

        # Initialize a list to accumulate the hinge loss for each batch
        Set_hinge_losssum = []
        
        # style_loss = self.style_content_loss(pred, ans, pred_size)
        
        for batch_ind in range(batch_size):
            item_loss = []

            for item_ind in range(num_items):
                true_class_label = c_label[true_indices[batch_ind]][item_ind]

                # Skip if the class label is 41
                if true_class_label != 41:
                    # Get indices where the class labels match the true class label
                    indices = tf.where(c_label == true_class_label)
                    # If there's only one match (i.e., the positive example itself)
                    if len(indices[:, 0]) == 1:
                        # Cosine similarity for the true positive
                        positive_score = cos_sim[batch_ind, :, item_ind, item_ind][true_indices[batch_ind]]
                        # Calculate loss but only positive ; negative is deemed as 0
                        item_loss.append(tf.reduce_sum(tf.maximum(0 - positive_score + slack_variable, 0.0)))
                        # item_loss.append(0)
                    else:
                        
                        # Cosine similarity for the true positive
                        positive_score = cos_sim[batch_ind, :, item_ind, item_ind][true_indices[batch_ind]]

                        # Cosine similarities for the negative examples
                        negative_scores = tf.stack([
                            cos_sim[batch_ind, indices[i, 0], item_ind, indices[i, 1]]
                            for i in range(indices.shape[0])
                            if indices[i, 0] != true_indices[batch_ind]
                        ])
                        
                        
                        # Calculate hinge loss for this item
                        item_loss.append(tf.reduce_sum(tf.maximum(negative_scores - positive_score + slack_variable, 0.0)))
                else:
                    item_loss.append(0)

            # Calculate the average loss for valid items
            filtered_values = [v for v in item_loss if isinstance(v, tf.Tensor)]
            if len(filtered_values) == 0:
                Set_hinge_losssum.append(0)
            else:
                Set_hinge_losssum.append(sum(filtered_values) / len(filtered_values))

        # Calculate the average loss across all batches
        hinge_Loss = sum(Set_hinge_losssum) / len(Set_hinge_losssum)
        
        # style loss + hinge loss
        Loss = 0.1 * hinge_Loss
        # Loss = style_loss

        return Loss
    # train step
    def train_step(self,data):
        # c2ラベル導入の注意点: 10001 => 0 のようなラベルmappingが必須
        # 抽出する予測候補は正解の位置のもの そのまま取り出すとクエリのものになるから
        # x = {x, x_size}, y_true : set label to identify positive pair. (nSet, )
        x, y_true = data
        # x : (nSet, nItemMax, dim) , x_size : (nSet, )
        if len(x) == 2:
            x, x_size = x
        else:
            x, x_size, c_label, c1_label = x 
        
        y_true = self.cross_set_label(y_true)
        y_true = tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))

        if not self.use_all_pred:
            # c_label (before): odd and even number index hasn't been shuffuled => c_label(after): each index indicates the answer
            ans_c_label = tf.gather(c_label, tf.where(tf.equal(y_true,1))[:,1])
            ans_c1_label = tf.gather(c1_label, tf.where(tf.equal(y_true,1))[:,1])
            pred_size = tf.gather(x_size, tf.where(tf.equal(y_true,1))[:,1])
        
        # tf.where(tf.equal(y_true,1))[:,1]
        # gallery : (nSet, nItemMax, dim)
        gallery = x
        # gallery linear projection(dimmension reduction) 
        if self.isTrainableMLP:
            gallery, _ = self.MLP(gallery, training=False)
        else:
            gallery = self.fc_cnn_proj(gallery) # : (nSet, nItemMax, d=baseChn*max_channel_ratio)

        # マッチングモデルを通した処理の記述など
        _, Real_score, _ = self.SetMatchingModel((gallery, x_size), training=False)
        Real_score = tf.reshape(Real_score, [gallery.shape[0], gallery.shape[0]])
        Real_score = tf.linalg.diag_part(tf.gather(Real_score,tf.where(tf.equal(y_true,1))[:,1]))
        with tf.GradientTape() as tape:
            # predict
            # predSMN : (nSet, nItemMax, d)
            if not self.use_all_pred:
                predCNN, predSMN, debug = self((x, x_size, ans_c_label, ans_c1_label, pred_size), training=True)
            else:
                predCNN, predSMN, debug = self((x, x_size), training=True)

            if not self.label_slice:
                predSMN = tf.gather(predSMN, ans_c1_label, batch_dims=1)
            
            # ---------------山園追加部分------------------

            tiled_gallery = tf.tile(tf.expand_dims(gallery, axis=1), [1,gallery.shape[0],1,1])
            # for文でペアになる部分をpredSMNで置換
            # 分かりやすいように対角にpairが来るようにしています Real_scoreはy_true参照の非対角の並び方
            for batch_ind in range(len(tiled_gallery)):
                tiled_gallery = tf.tensor_scatter_nd_update(tiled_gallery, [[batch_ind, batch_ind]],[predSMN[batch_ind]])
            _, Fake_score, _ = self.SetMatchingModel((tiled_gallery, x_size), training=False)
            Fake_score = tf.reshape(Fake_score, [gallery.shape[0], gallery.shape[0]])
            Fake_score = tf.linalg.diag_part(Fake_score)
            
            # --------------------------------------------
            setMatchingloss = self.SetMatchingScore_loss(Real_score, Fake_score)
            # compute similairty with gallery and f1_bert_score
            # input gallery as x and predSMN as y in each bellow set similarity function. 

            if not self.use_all_pred :
                cos_sim_before = self.cos_similarity((gallery, predSMN), PIFR=False)
                # cos_sim = self.cos_similarity((weights_gallery, predSMN), PIFR=True)
            elif self.calc_set_sim == 'CS':
                set_score = self.cross_set_score((gallery, predSMN), x_size)
            elif self.calc_set_sim == 'BERTscore':
                set_score = self.BERT_set_score((gallery, predSMN), x_size)
            else:
                print("指定された集合間類似度を測る関数は存在しません")
                sys.exit()
            
            # cos_sim loss related
            if self.cos_sim_method == 'CE':
                cos_sim_loss = 0
            elif self.cos_sim_method == 'Hinge':
                cos_sim_loss = self.item_hinge_loss(cos_sim=cos_sim_before, y_true=y_true, c_label=c1_label, pred=predSMN, ans=tf.gather(gallery, tf.where(tf.equal(y_true,1))[:,1]), pred_size=pred_size)
            else:
                # PIFR function - PIFR accuracy
                weights = self.compute_representation_weights(predSMN, gallery, x_size)
                # y_predごとにgalleryの各集合をPIFR weightで再構成
                weights_gallery = [tf.matmul(weights[i][j], gallery[j]) for i in range(predSMN.shape[0]) for j in range(gallery.shape[0])] 
                weights_gallery = tf.reshape(weights_gallery, [predSMN.shape[0], gallery.shape[0], gallery.shape[1], gallery.shape[2]])

                cos_sim_after = self.cos_similarity((weights_gallery, predSMN), PIFR=True)
                PIFR_score = tf.reduce_sum(tf.linalg.diag_part(cos_sim_after),axis=-1)/ tf.expand_dims(pred_size,axis=-1)
                cos_sim_loss = self.PIFR_Loss(PIFR_score, y_true, Loss_method='Hinge')
            # style loss related
            if self.style_method == 'item_style':
                style_loss = self.style_content_loss(predSMN, gallery, pred_size)
            elif self.style_method == 'DFA_style':
                if self.cos_sim_method != 'DFA':
                    # PIFR function - PIFR accuracy
                    weights = self.compute_representation_weights(predSMN, gallery, x_size)
                    # y_predごとにgalleryの各集合をPIFR weightで再構成
                    weights_gallery = [tf.matmul(weights[i][j], gallery[j]) for i in range(predSMN.shape[0]) for j in range(gallery.shape[0])] 
                    weights_gallery = tf.reshape(weights_gallery, [predSMN.shape[0], gallery.shape[0], gallery.shape[1], gallery.shape[2]])
                    style_loss = self.style_content_loss(predSMN, weights_gallery, pred_size)
                else:
                    style_loss = self.style_content_loss(predSMN, weights_gallery, pred_size)
            
            if not self.use_all_pred:
                if self.is_c1label:
                    # c1_label_tiled = tf.tile(tf.expand_dims(c1_label,axis=0), (len(c1_label),1,1))
                    loss = self.compiled_loss(y_pred = cos_sim_before, y_true = c1_label, regularization_losses=self.losses) + cos_sim_loss + style_loss + setMatchingloss
                else:
                    loss = self.compiled_loss(y_pred = cos_sim, y_true = c_label, regularization_losses=self.losses)
            else:
                loss = self.compiled_loss(set_score, y_true, regularization_losses=self.losses)
        # train using gradients
        trainable_vars = self.trainable_variables
        
        
        # train parameters excepts for CNN
        trainable_vars = [v for v in trainable_vars if 'cnn' not in v.name]
        trainable_vars = [v for v in trainable_vars if 'set_matching_model' not in v.name]
       
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, trainable_vars)
            if grad is not None)

        # update metrics

        if not self.use_all_pred:
            '''
            if self.is_c1label:
                # c1_label_tiled = tf.tile(tf.expand_dims(c1_label,axis=0), (len(c1_label),1,1))
                custom_metric_value = self.set_retrieval_rank(cos_sim=cos_sim_before, y_true=y_true, c_label=c1_label)
            else:
                custom_metric_value = self.set_retrieval_rank(cos_sim, y_true, c_label)
            '''
            self.compiled_metrics.update_state(y_true, Fake_score)
            # category accuracyの導入
            # custom_metric_valueで最近傍ベクトルのラベルを返せれば
            # pred_label = self.return_pred_label(cos_sim_before, c1_label)
            # self.compiled_metrics.update_state(pred_label, ans_c1_label)
            # return metrics as dictionary
            return {'Match_loss': setMatchingloss, 'style_loss': style_loss, 'Set_accuracy':self.metrics[1].result()}
        else:
            self.compiled_metrics.update_state(set_score, y_true)
            # return metrics as dictionary
            return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):

        x, y_true = data
        # x : (nSet, nItemMax, dim) , x_size : (nSet, )
        if len(x) == 2:
            x, x_size = x
        else:
            x, x_size, c_label, c1_label = x 
        #cross set label creation
        # y_true : [(1,0...,0),(0,1,...,0),...,(0,0,...,1)] locates where the positive is. (nSet, nSet)
        y_true = self.cross_set_label(y_true)
        y_true = tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))

        if not self.use_all_pred:
            # c_label (before): odd and even number index hasn't been shuffuled => c_label(after): each index indicates the answer
            ans_c_label = tf.gather(c_label, tf.where(tf.equal(y_true,1))[:,1])
            ans_c1_label = tf.gather(c1_label, tf.where(tf.equal(y_true,1))[:,1])
            pred_size = tf.gather(x_size, tf.where(tf.equal(y_true,1))[:,1])
        
        # gallery : (nSet, nItemMax, dim)
        gallery = x 

        # gallery linear projection(dimmension reduction) 
        if self.isTrainableMLP:
            gallery, _ = self.MLP(gallery, training=False)
        else:
            gallery = self.fc_cnn_proj(gallery) # : (nSet, nItemMax, d=baseChn*max_channel_ratio)

        # predict
        # predSMN : (nSet, nItemMax, d)
        if not self.use_all_pred:
            predCNN, predSMN, debug = self((x, x_size, ans_c_label, ans_c1_label, pred_size), training=False)
        else:
            predCNN, predSMN, debug = self((x, x_size), training=False)
        
        if not self.label_slice:
            predSMN = tf.gather(predSMN, ans_c1_label, batch_dims=1)

        # ---------------山園追加部分------------------
        # マッチングモデルを通した処理の記述など
        _, Real_score, _ = self.SetMatchingModel((gallery, x_size), training=False)
        Real_score = tf.reshape(Real_score, [gallery.shape[0], gallery.shape[0]])
        Real_score = tf.linalg.diag_part(tf.gather(Real_score,tf.where(tf.equal(y_true,1))[:,1]))

        tiled_gallery = tf.tile(tf.expand_dims(gallery, axis=1), [1,gallery.shape[0],1,1])
        # for文でペアになる部分をpredSMNで置換
        # 分かりやすいように対角にpairが来るようにしています Real_scoreはy_true参照の非対角の並び方
        for batch_ind in range(len(tiled_gallery)):
            tiled_gallery = tf.tensor_scatter_nd_update(tiled_gallery, [[batch_ind, batch_ind]],[predSMN[batch_ind]])
        _, Fake_score, _ = self.SetMatchingModel((tiled_gallery, x_size), training=False)
        Fake_score = tf.reshape(Fake_score, [gallery.shape[0], gallery.shape[0]])
        Fake_score = tf.linalg.diag_part(Fake_score)

        # --------------------------------------------
        setMatchingloss = self.SetMatchingScore_loss(Real_score, Fake_score)
        # --------------------------------------------
        if not self.use_all_pred :
            cos_sim_before = self.cos_similarity((gallery, predSMN), PIFR=False)
            # cos_sim = self.cos_similarity((weights_gallery, predSMN), PIFR=True)
        elif self.calc_set_sim == 'CS':
            set_score = self.cross_set_score((gallery, predSMN), x_size)
        elif self.calc_set_sim == 'BERTscore':
            set_score = self.BERT_set_score((gallery, predSMN), x_size)
        else:
            print("指定された集合間類似度を測る関数は存在しません")
            sys.exit()
        
        # cos_sim loss related
        if self.cos_sim_method == 'CE':
            cos_sim_loss = 0
        elif self.cos_sim_method == 'Hinge':
            cos_sim_loss = self.item_hinge_loss(cos_sim=cos_sim_before, y_true=y_true, c_label=c1_label, pred=predSMN, ans=tf.gather(gallery, tf.where(tf.equal(y_true,1))[:,1]), pred_size=pred_size)
        else:
            # PIFR function - PIFR accuracy
            weights = self.compute_representation_weights(predSMN, gallery, x_size)
            # y_predごとにgalleryの各集合をPIFR weightで再構成
            weights_gallery = [tf.matmul(weights[i][j], gallery[j]) for i in range(predSMN.shape[0]) for j in range(gallery.shape[0])] 
            weights_gallery = tf.reshape(weights_gallery, [predSMN.shape[0], gallery.shape[0], gallery.shape[1], gallery.shape[2]])

            cos_sim_after = self.cos_similarity((weights_gallery, predSMN), PIFR=True)
            PIFR_score = tf.reduce_sum(tf.linalg.diag_part(cos_sim_after),axis=-1)/ tf.expand_dims(pred_size,axis=-1)
            cos_sim_loss = self.PIFR_Loss(PIFR_score, y_true, Loss_method='Hinge')
        # style loss related
        if self.style_method == 'item_style':
            style_loss = self.style_content_loss(predSMN, gallery, pred_size)
        elif self.style_method == 'DFA_style':
            if self.cos_sim_method != 'DFA':
                # PIFR function - PIFR accuracy
                weights = self.compute_representation_weights(predSMN, gallery, x_size)
                # y_predごとにgalleryの各集合をPIFR weightで再構成
                weights_gallery = [tf.matmul(weights[i][j], gallery[j]) for i in range(predSMN.shape[0]) for j in range(gallery.shape[0])] 
                weights_gallery = tf.reshape(weights_gallery, [predSMN.shape[0], gallery.shape[0], gallery.shape[1], gallery.shape[2]])
                style_loss = self.style_content_loss(predSMN, weights_gallery, pred_size)
            else:
                style_loss = self.style_content_loss(predSMN, weights_gallery, pred_size)
        if not self.use_all_pred:
            if self.is_c1label:
                # c1_label_tiled = tf.tile(tf.expand_dims(c1_label,axis=0), (len(c1_label),1,1))
                # loss = self.compiled_loss(y_pred = cos_sim, y_true = c1_label, regularization_losses=self.losses)
                loss = self.compiled_loss(y_pred = cos_sim_before, y_true = c1_label, regularization_losses=self.losses) + cos_sim_loss + style_loss +setMatchingloss 
            else:
                loss = self.compiled_loss(y_pred = cos_sim, y_true = c_label, regularization_losses=self.losses)
        else:
            loss = self.compiled_loss(set_score, y_true, regularization_losses=self.losses)

        if not self.use_all_pred:
            '''
            if self.is_c1label:
                # c1_label_tiled = tf.tile(tf.expand_dims(c1_label,axis=0), (len(c1_label),1,1))
                custom_metric_value = self.set_retrieval_rank(cos_sim_before, y_true, c1_label)
            else:
                custom_metric_value = self.set_retrieval_rank(cos_sim, y_true, c_label)
            '''
            # self.compiled_metrics.update_state(y_true, custom_metric_value)
            self.compiled_metrics.update_state(y_true, Fake_score)
            # pred_label = self.return_pred_label(cos_sim_before, c1_label)
            # self.compiled_metrics.update_state(pred_label, ans_c1_label)
            # return metrics as dictionary
            return {'Match_loss': setMatchingloss, 'style_loss': style_loss, 'Set_accuracy':self.metrics[1].result()}
            # return {m.name: m.result() for m in self.metrics}
        else:
            # update metrics
            self.compiled_metrics.update_state(set_score, y_true)
            # return metrics as dictionary
            return {m.name: m.result() for m in self.metrics}

    # predict step
    def predict_step(self,data):
        batch_data = data[0]
        # x = {x, x_size}, y_true : set label to identify positive pair. (nSet, )
        x, x_size, c_label, y_test, item_label  = batch_data
        
        #cross set label creation
        # y_true : [(1,0...,0),(0,1,...,0),...,(0,0,...,1)] locates where the positive is. (nSet, nSet)
        y_true = self.cross_set_label(y_test)
        y_true = tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))

        if not self.use_all_pred:
            # c_label (before): odd and even number index hasn't been shuffuled => c_label(after): each index indicates the answer
            ans_c_label = tf.gather(c_label, tf.where(tf.equal(y_true,1))[:,1])
            pred_size = tf.gather(x_size, tf.where(tf.equal(y_true,1))[:,1])
        

        # gallery : (nSet, nItemMax, dim)
        gallery = x
        # gallery linear projection(dimmension reduction) 
        if self.isTrainableMLP:
            gallery, _ = self.MLP(gallery, training=False)
        else:
            gallery = self.fc_cnn_proj(gallery) # : (nSet, nItemMax, d=baseChn*max_channel_ratio)
        # predict
        # predSMN : (nSet, nItemMax, d)
        predCNN, predSMN, debug = self((x, x_size, [], ans_c_label, pred_size), training=False)

        if not self.label_slice:
            predSMN = tf.gather(predSMN, ans_c_label, batch_dims=1)
        
        if len(x) <= 100:
            # マッチングモデルを通した処理の記述など
            _, Real_score, _ = self.SetMatchingModel((gallery, x_size), training=False)
            Real_score = tf.reshape(Real_score, [gallery.shape[0], gallery.shape[0]])
            Real_score = tf.linalg.diag_part(tf.gather(Real_score,tf.where(tf.equal(y_true,1))[:,1]))
            # Matching scoreはミニバッチごとに行わないとメモリ不足になる
            tiled_gallery = tf.tile(tf.expand_dims(gallery, axis=1), [1,gallery.shape[0],1,1])
            # for文でペアになる部分をpredSMNで置換
            # 分かりやすいように対角にpairが来るようにしています Real_scoreはy_true参照の非対角の並び方
            for batch_ind in range(len(tiled_gallery)):
                tiled_gallery = tf.tensor_scatter_nd_update(tiled_gallery, [[batch_ind, batch_ind]],[predSMN[batch_ind]])
            _, Fake_score, _ = self.SetMatchingModel((tiled_gallery, x_size), training=False)
            Fake_score = tf.reshape(Fake_score, [gallery.shape[0], gallery.shape[0]])
            Fake_score = tf.linalg.diag_part(Fake_score)
        
        
        set_label = tf.cast(y_test, tf.int64)
        replicated_set_label = tf.tile(tf.expand_dims(set_label, axis=1), [1, len(x[0])])
        query_id = tf.stack([replicated_set_label, item_label],axis=1)
        query_id = tf.transpose(query_id, [0,2,1])
        if len(x) <= 100:
            return predSMN, gallery, replicated_set_label, query_id, Real_score, Fake_score
        else:
            return predSMN, gallery, replicated_set_label, query_id
#----------------------------
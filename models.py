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
from layers import layer_normalization, cross_set_score, set_attention
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense

warnings.filterwarnings("ignore", category=ConvergenceWarning)
#----------------------------

# normalization
# to => layers.py
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
# set matching network
class SMN(tf.keras.Model):
    def __init__(self, is_set_norm=False, num_layers=1, num_heads=2, calc_set_sim='BERTscore', baseChn=32, baseMlp = 512, rep_vec_num=1, seed_init = 0, max_channel_ratio=2, set_loss=False, is_category_emb=False, c1_label=True, gallerytype = 'OutBatch', style_loss = 'item_style', L2_norm_loss = False, whitening=None):
        super(SMN, self).__init__()
        self.num_layers = num_layers
        self.calc_set_sim = calc_set_sim
        self.rep_vec_num = rep_vec_num
        self.seed_init = seed_init
        self.baseChn = baseChn
        self.baseMlpChn = baseMlp
        self.set_loss = set_loss
        self.is_category_emb = is_category_emb
        self.is_c1label = c1_label
        self.gallerytype = gallerytype
        self.style_method = style_loss
        self.is_L2_norm_loss = L2_norm_loss
        self.linear_parampath = whitening
        self.isSelf = True
        self.isCross = True
        self.isSelfShareParam = False
        self.isCrossShareParam = False
        self.l2_loss_weight = 1.0
        isInstanceNorm = False
        isCrossNorm = False
        
        
        if self.seed_init != 0:
            self.dim_shift15 = len(self.seed_init[0])
        
        #---------------------
        # Trainable fc projection
        self.fc_cnn_proj = tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching')
        #---------------------
        # projection layer for pred
        self.fc_pred_proj = tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn') # nameにcnn
        #---------------------
        # cnn model
        self.x_fc_cnn_proj = Dense(baseChn*max_channel_ratio,activation=tfa.activations.gelu,use_bias=False,name='setmatching')
        self.y_fc_cnn_proj = Dense(baseChn*max_channel_ratio,activation=tfa.activations.gelu,use_bias=False,name='setmatching')

        ## layers in encoder
        # sharing parameters between x and y
        self.self_attentions = [ set_attention(head_size=baseChn*max_channel_ratio,num_heads=num_heads,self_attention=True) for i in range(num_layers) ]
        self.layer_norms_enc1 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers)]
        self.layer_norms_enc2 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers)]
        self.fcs_enc = [ Dense(baseChn*max_channel_ratio,activation=tfa.activations.gelu,use_bias=False,name='setmatching_fcs_enc') for i in range(num_layers) ]
        # x
        self.x_self_attentions = [ set_attention(head_size=baseChn*max_channel_ratio,num_heads=num_heads,self_attention=True) for i in range(num_layers) ]
        self.x_layer_norms_enc1 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers)]
        self.x_layer_norms_enc2 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers)]
        self.x_fcs_enc = [ Dense(baseChn*max_channel_ratio,activation=tfa.activations.gelu,use_bias=False,name='setmatching_x_fcs_enc') for i in range(num_layers) ]
        # y
        self.y_self_attentions = [ set_attention(head_size=baseChn*max_channel_ratio,num_heads=num_heads,self_attention=True) for i in range(num_layers) ]
        self.y_layer_norms_enc1 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers)]
        self.y_layer_norms_enc2 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers)]
        self.y_fcs_enc = [ Dense(baseChn*max_channel_ratio,activation=tfa.activations.gelu,use_bias=False,name='setmatching_y_fcs_enc') for i in range(num_layers) ]

        ## layers in decoder
        # sharing parameters between x and y
        self.cross_attentions = [ set_attention(head_size=baseChn*max_channel_ratio,num_heads=num_heads) for i in range(num_layers) ]
        self.layer_norms_dec1 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers) ]
        self.layer_norms_dec2 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers) ]
        self.fcs_dec = [ Dense(baseChn*max_channel_ratio,activation=tfa.activations.gelu,use_bias=False,name='setmatching_fcs_dec') for i in range(num_layers) ]
        # x
        self.x_cross_attentions = [ set_attention(head_size=baseChn*max_channel_ratio,num_heads=num_heads) for i in range(num_layers) ]
        self.x_layer_norms_dec1 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers) ]
        self.x_layer_norms_dec2 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers) ]
        self.x_fcs_dec = [ Dense(baseChn*max_channel_ratio,activation=tfa.activations.gelu,use_bias=False,name='setmatching_x_fcs_dec') for i in range(num_layers) ]
        # y
        self.y_cross_attentions = [ set_attention(head_size=baseChn*max_channel_ratio,num_heads=num_heads) for i in range(num_layers) ]
        self.y_layer_norms_dec1 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers) ]
        self.y_layer_norms_dec2 = [ layer_normalization(size_d=baseChn*max_channel_ratio,is_instance_norm=isInstanceNorm,is_cross=isCrossNorm) for i in range(num_layers) ]
        self.y_fcs_dec = [ Dense(baseChn*max_channel_ratio,activation=tfa.activations.gelu,use_bias=False,name='setmatching_y_fcs_dec') for i in range(num_layers) ]

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
        self.key_cluster = False
        #---------------------

        # マッチングモデルの初期化定義
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

        # whitening path load
        with open(self.linear_parampath, 'rb') as fp:
            self.gause_noise = pickle.load(fp)
            self.whitening_mean = pickle.load(fp)
            self.whitening_std = pickle.load(fp)
        
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
    
    def call(self, inputs):
        debug = {}
        # shape = tf.shape(x)
        # nSet = shape[0]
        # nItemMax = shape[1]
        
        X,Y = inputs
        x,x_size = X
        y,y_size = Y
        x_size = tf.squeeze(x_size)
        y_size = tf.squeeze(y_size)

        x_shape = tf.shape(x)
        y_shape = tf.shape(y)
        x_nSet = x_shape[0]
        y_nSet = y_shape[0]
        x_nItemMax = x_shape[1]
        y_nItemMax = y_shape[1]
        debug = {}
        
        # category embeddeing
        # if self.is_category_emb and not self.set_loss:
        #     x  += tf.gather(tf.stack(self.category_emb), c_label)
        # CNN
        predCNN = []
        
        debug['x_encoder_layer_0'] = x

        x_2enc = x
        #---------------------
        
        x = tf.tile(tf.expand_dims(x,1),[1,y_nSet,1,1])
        pre_y = y
        y = tf.tile(tf.expand_dims(y,1),[1,x_nSet,1,1])
        #---------------------
        #---------------------------------------------------------------------
        # encoder (self-attention)
        if self.isSelf:
            for i in range(self.num_layers):
                if self.isSelfShareParam:
                    # layer normalization
                    z_x = self.layer_norms_enc1[i](x,x_size,y,y_size)
                    z_y = self.layer_norms_enc1[i](y,y_size,x,x_size)

                    # self-attention  input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
                    z_x = self.self_attentions[i](z_x,z_x)
                    z_y = self.self_attentions[i](z_y,z_y)

                    # skip connection
                    x += z_x
                    y += z_y

                    # layer normalization
                    z_x = self.layer_norms_enc2[i](x,x_size,y,y_size)
                    z_y = self.layer_norms_enc2[i](y,y_size,x,x_size)

                    # fully-connected layer
                    z_x = self.fcs_enc[i](z_x)
                    z_y = self.fcs_enc[i](z_y)

                    # skip connection
                    x += z_x
                    y += z_y

                    debug[f'x_encoder_layer_{i+1}'] = x
                    debug[f'y_encoder_layer_{i+1}'] = y
                    

                else:
                    
                    # layer normalization
                    z_x = self.x_layer_norms_enc1[i](x,x_size,y,y_size)
                    z_y = self.y_layer_norms_enc1[i](y,y_size,x,x_size)

                    # self-attention  input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
                    z_x = self.x_self_attentions[i](z_x,z_x)
                    z_y = self.y_self_attentions[i](z_y,z_y)

                    # skip connection
                    x += z_x
                    y += z_y

                    # layer normalization
                    z_x = self.x_layer_norms_enc2[i](x,x_size,y,y_size)
                    z_y = self.y_layer_norms_enc2[i](y,y_size,x,x_size)

                    # fully-connected layer
                    z_x = self.x_fcs_enc[i](z_x)
                    z_y = self.y_fcs_enc[i](z_y)

                    # skip connection
                    x += z_x
                    y += z_y

                    debug[f'x_encoder_layer_{i+1}'] = x
                    debug[f'y_encoder_layer_{i+1}'] = y

        x_enc = x
        y_enc = y

         # decoder (cross-attention)
        if self.isCross:
            for i in range(self.num_layers):
                if self.isCrossShareParam:
                    
                    # pivot attention
                    self.cross_attentions[i].cseft_with_self = True
                    self.cross_attentions[i].cseft_with_self = True

                    # layer normalization
                    z_x = self.layer_norms_dec1[i](x,x_size,y,y_size)
                    z_y = self.layer_norms_dec1[i](y,y_size,x,x_size)

                    # cross-attention  input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
                    z_x = self.cross_attentions[i](z_x,z_y)
                    z_y = self.cross_attentions[i](z_y,z_x)

                    # skip connection
                    x += z_x
                    y += z_y

                    # layer normalization
                    z_x = self.layer_norms_dec2[i](x,x_size,y,y_size)
                    z_y = self.layer_norms_dec2[i](y,y_size,x,x_size)

                    # fully-connected layer
                    z_x = self.fcs_dec[i](z_x)
                    z_y = self.fcs_dec[i](z_y)

                    # skip connection
                    x += z_x
                    y += z_y

                    debug[f'x_decoder_layer_{i+1}'] = x
                    debug[f'y_decoder_layer_{i+1}'] = y


                else:
                    # pivot attention
                    self.x_cross_attentions[i].cseft_with_self = True
                    self.y_cross_attentions[i].cseft_with_self = True
                    # layer normalization
                    z_x = self.x_layer_norms_dec1[i](x,x_size,y,y_size)
                    z_y = self.y_layer_norms_dec1[i](y,y_size,x,x_size)

                    # cross-attention  input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
                    z_x = self.x_cross_attentions[i](z_x,z_y)
                    z_y = self.y_cross_attentions[i](z_y,z_x)

                    # skip connection
                    x += z_x
                    y += z_y

                    # layer normalization
                    z_x = self.x_layer_norms_dec2[i](x,x_size,y,y_size)
                    z_y = self.y_layer_norms_dec2[i](y,y_size,x,x_size)

                    # fully-connected layer
                    z_x = self.x_fcs_dec[i](z_x)
                    z_y = self.y_fcs_dec[i](z_y)

                    # skip connection
                    x += z_x
                    y += z_y

                    debug[f'x_decoder_layer_{i+1}'] = x
                    debug[f'y_decoder_layer_{i+1}'] = y

        x_dec = x
        y_dec = y

        diag_indices = tf.range(y.shape[0])
        indices = tf.stack([diag_indices, diag_indices], axis=1)  # Shape: [20, 2]

        # 対角成分を抽出
        y_pred = tf.gather_nd(y, indices)

        #---------------------------------------------------------------------
        
        return pre_y, y_pred, debug
    
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
            if self.set_loss:
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
            x = tf.cast(x, dtype=tf.float32)
            x_expand = tf.expand_dims(tf.nn.l2_normalize(x, axis=-1), 0) # (nSet_x, 1, nItemMax, head_size)
            y_expand = tf.expand_dims(tf.nn.l2_normalize(y, axis=-1), 1) # (1, nSet_y, nItemMax, head_size)
            cos_sim = tf.einsum('aijk,ibmk->abjm', y_expand, x_expand) # (nSet_y, nSet_x, nItemMax_x, nItemMax_y)
        else:
            x, y = x # x, y : (nSet_x(y), nItemMax, dim)
            cos_sim = tf.stack([tf.einsum('ik,jbk->jib', tf.nn.l2_normalize(y[i], axis=-1), tf.nn.l2_normalize(x[i], axis=-1)) for i in range(len(x))])
            
        return cos_sim
    
    def cos_similarity_pos_neg(self, x, gallerytype='Inbatch'):
        x, y = x # x, y : (nSet_x(y), nItemMax, dim)
        # x : Gallery (batch, nItemMax, nPositive(1) + nNegativeMax(30), D)
        # y : Pred (batch, nItemMax, D)
        x = tf.cast(x, dtype=tf.float32)
        if gallerytype == 'InBatch':
            x_expand = tf.expand_dims(tf.nn.l2_normalize(x, axis=-1), 0) # (nSet_x, 1, nItemMax, head_size)
            y_expand = tf.expand_dims(tf.nn.l2_normalize(y, axis=-1), 1) # (1, nSet_y, nItemMax, head_size)
            cos_sim = tf.einsum('aijk,ibmk->abjm', y_expand, x_expand) # (nSet_y, nSet_x, nItemMax_x, nItemMax_y)
        else:
            x_normalized = tf.nn.l2_normalize(x, axis=-1)  # x: [nSet_x, nItemMax, nPositive + nNegative, D]
            y_normalized = tf.nn.l2_normalize(y, axis=-1)  # y: [nSet_y, nItemMax, D], nSet_x == nSet_y
            cos_sim = tf.einsum('bfqc,bfc->bfq', x_normalized, y_normalized)  # Shape: [nSet_y, nItemMax, nPositive + nNegative]
        
        return cos_sim
    
    def gram_matrix(self, input_tensor, is_gallery=False):
        if not is_gallery:
            # result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
            result = tf.einsum('bnd,bne->bnde', input_tensor, input_tensor) # pixel loss
            # result = tf.einsum('bnd,bnd->bn', input_tensor, input_tensor)
            input_shape = tf.shape(input_tensor)
            num_locations = tf.cast(input_shape[-1]*input_shape[-1], tf.float32)
        else:
            # result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
            result = tf.einsum('bfqd,bfqe->bfqde', input_tensor, input_tensor) # pixel loss
            # result = tf.einsum('bnd,bnd->bn', input_tensor, input_tensor)
            input_shape = tf.shape(input_tensor)
            num_locations = tf.cast(input_shape[-1]*input_shape[-1], tf.float32)
        return result/(num_locations)
    
    def style_content_loss(self, pred, ans, pred_size):
        # style_outputs = outputs['style']
        # content_outputs = outputs['content']
        # style pixel loss ver.
        pred_gram = self.gram_matrix(pred)
        ans = tf.cast(ans, tf.float32)
        if self.style_method == 'item_style':
            ans_gram = self.gram_matrix(ans, is_gallery=False)
        elif self.style_method == 'DFA_style':
            ans_gram = tf.stack([self.gram_matrix(ans[i]) for i in range(len(pred))]) # PIFR ver
            ans_gram = tf.stack([ans_gram[i,i,:,:,:] for i in range(len(pred))])
        
        gram_shape = pred_gram.shape
        pred_gram = tf.reshape(pred_gram, [gram_shape[0], gram_shape[1], gram_shape[2]*gram_shape[3]])
        ans_gram = tf.reshape(ans_gram, [gram_shape[0], gram_shape[1], gram_shape[2]*gram_shape[3]])
        
        # item_style_loss = tf.reduce_sum(tf.reduce_mean(tf.square(pred_gram-ans_gram),axis=-1)) / pred_size
        # Gram_dist = tf.square(pred_gram-ans_gram)
        # Gram_dist = tf.reshape(Gram_dist, [pred.shape[0], pred.shape[1], pred.shape[2]*pred.shape[2]])
        item_style_loss = tf.reduce_sum(tf.reduce_mean(tf.square(pred_gram-ans_gram),axis=-1), axis=-1) / pred_size
        # item_style_loss = tf.reduce_sum(tf.reduce_mean(tf.square(pred_gram-ans_gram),axis=-1), axis=-1) / pred_size
        
        # item_style_loss = tf.reduce_sum(tf.square(self.gram_matrix(pred)-self.gram_matrix(ans)),axis=-1) / pred_size
        style_loss = tf.reduce_mean(item_style_loss)

        return style_loss

    def contrastive_style_loss(self, pred, ans, pred_size, margin=1.0):
        # style_outputs = outputs['style']
        # content_outputs = outputs['content']
        # style pixel loss ver.
        
        range_items = tf.range(pred.shape[1], dtype=tf.float32)  # (nItemMax,)
        item_mask = tf.expand_dims(pred_size, axis=1) > range_items  # Shape: (Batch, nItemMax)
        pred_gram = self.gram_matrix(pred)
        ans = tf.cast(ans, tf.float32)
        
        ans_gram = self.gram_matrix(ans, is_gallery=True)

        # Positiveスコア (Ansの[batch, item, 0]とPredの2乗誤差)
        positive_gram = ans_gram[:, :, 0, :, :]  # Shape: [Batch, Item, D, D]
        positive_scores = tf.reduce_mean(tf.square(pred_gram - positive_gram), axis=[-2, -1])  # Shape: [Batch, Item]

        # Negativeスコア (Ansの[batch, item, 1:]とPredの2乗誤差)
        negative_gram = ans_gram[:, :, 1:, :, :]  # Shape: [Batch, Item, P_Nnum-1, D, D]
        # negative_scores = tf.reduce_mean(tf.square(tf.expand_dims(pred_gram, axis=2) - negative_gram), axis=[-2, -1])  # Shape: [Batch, Item, P_Nnum-1]

        diff = np.expand_dims(pred_gram, axis=2) - negative_gram  # Shape: [Batch, Item, P_Nnum-1, D, D]
        squared_diff = np.square(diff)  # 差分の2乗: Shape [Batch, Item, P_Nnum-1, D, D]
        mean_squared_diff = np.mean(squared_diff, axis=(-2, -1))  # 最後の2次元(D, D)を平均: Shape [Batch, Item, P_Nnum-1]

        # ヒンジ損失計算
        # margin - positive_score + negative_score の最大値を計算
        item_hinge_loss = tf.nn.relu(tf.expand_dims(positive_scores, axis=-1) - mean_squared_diff + margin)  # Shape: [Batch, Item, P_Nnum-1]

        item_loss = tf.reduce_mean(item_hinge_loss, axis=-1)

        masked_item_loss = item_loss * tf.cast(item_mask, tf.float32)

        batch_loss = tf.reduce_sum(masked_item_loss, axis=1) / tf.maximum(pred_size, 1.0)  # Shape: (Batch,)

        return tf.reduce_mean(batch_loss)
    
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
            bce = tf.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
            result = bce(y_true=y_true, y_pred=score)
        elif Loss_method == 'Hinge':
            # Yが1の部分をマスクとして使い、ポジティブスコアを取得
            positive_scores = tf.reduce_sum(score * y_true, axis=1)  # 各行のポジティブスコア (shape: [batch_size])

            # Yが0の部分をマスクして、ネガティブスコアを取得
            negative_scores = score * (1 - y_true)  # 各行でYが0のスコア (shape: [batch_size, num_classes])

            # ポジティブスコアをネガティブスコアにブロードキャストし、max(negative - positive + slack, 0.0) を計算
            hinge_loss = tf.maximum(negative_scores - tf.expand_dims(positive_scores, axis=1) + 0.2, 0.0)

            # 各行ごとのヒンジ損失を合計
            result = tf.reduce_sum(hinge_loss, axis=1)
        
        return tf.reduce_mean(result)
    
    # L2 norm loss between pred and positive set
    # * pred<= predSMN, y<= before_pred
    def L2_norm_loss(self, pred, y, pred_size):
        # pred, y : (Batch, nItemMax, Dim)
        batch_size, item_size, dim = pred.shape
        y = tf.cast(y, tf.float32)

        # cos_sim related striction
        # norm_ratio = (tf.norm(pred,axis=-1) + 1e-6) / (tf.norm(y, axis=-1) + 1e-6)
        # L2_loss = tf.nn.relu(-(1.0 - norm_ratio))
        range_items = tf.range(item_size, dtype=tf.float32)  # (nItemMax,)
        item_mask = tf.expand_dims(pred_size, axis=1) > range_items  # Shape: (Batch, nItemMax)
        # pred = (pred-self.whitening_mean) / self.whitening_std
        # y = (y-self.whitening_mean) / self.whitening_std

        # normalization
        # pred = tf.nn.l2_normalize(pred, axis=1)
        # y = tf.nn.l2_normalize(y, axis=1)

        L2_diff = tf.norm(pred - y, axis=-1) # (Batch, nItemMax)
        L2_diff = L2_diff * self.l2_loss_weight

        masked_item_loss = L2_diff * tf.cast(item_mask, tf.float32)
        # masked_item_loss = L2_loss * tf.cast(item_mask, tf.float32)
        L2_loss = tf.reduce_sum(masked_item_loss, axis=-1) / tf.maximum(pred_size, 1.0) # (Batch,)

        return tf.reduce_mean(L2_loss)
    # train step
    def train_step(self,data):
        # x = {x, x_size}, y_true : set label to identify positive pair. (nSet, )
        x, y_true = data
        # x : (nSet, nItemMax, dim) , x_size : (nSet, )
        # negative : (nSet, nItemMax, nNegative, dim)
        if len(x) == 2:
            x, x_size = x
        else:
            x, x_size, c_label, negative = x  
        
        y_true = self.cross_set_label(y_true)
        y_true = tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))

        if not self.set_loss:
            # c_label (before): odd and even number index hasn't been shuffuled => c_label(after): each index indicates the answer
            ans_c_label = tf.gather(c_label, tf.where(tf.equal(y_true,1))[:,1])
            pred_size = tf.gather(x_size, tf.where(tf.equal(y_true,1))[:,1])

        # whitening_step
        if self.linear_parampath != None:
            x = np.matmul(x, self.gause_noise) # equals to fc(x) 
            # x = (x- self.whitening_mean) / self.whitening_std # normalization
            x = tf.constant(x)
        else:
            x = self.fc_cnn_proj(x)
        # gallery : (nSet, nItemMax, dim)
        gallery = x

        # negative gallery + positive gallery generating
        pos_G = tf.gather(gallery,tf.where(tf.equal(y_true,1))[:,1])
        pos_G = tf.expand_dims(pos_G, axis=2)

        if self.linear_parampath != None:
            negative = np.matmul(negative, self.gause_noise)
        else:
            negative = self.fc_cnn_proj(negative)
        # neg_G = tf.gather(negative, ans_c_label)
        pos_neg_G = tf.concat([pos_G, negative],axis=2)

    
        with tf.GradientTape() as tape:
            # predict
            # predSMN : (nSet, nItemMax, d)
            # y_pred = tf.tile(self.set_emb, [x.shape[0], 1,1]) # random init ver
            y_pred = tf.tile(tf.expand_dims(tf.stack(self.set_emb), axis=0),[x.shape[0],1,1])
            y_pred = tf.matmul(y_pred, tf.cast(self.gause_noise, tf.float32))
            y_pred = tf.gather(y_pred, ans_c_label, batch_dims=1)
            
            if not self.set_loss:
                # predCNN, predSMN, debug = self((x, x_size, ans_c_label, ans_c1_label, pred_size), training=True)
                before_pred, predSMN, debug = self(((x, x_size), (y_pred, pred_size)), training=True)
            else:
                before_pred, predSMN, debug = self((x, x_size), training=True)
            
            # if not self.label_slice:
            #     predSMN = tf.gather(predSMN, ans_c_label, batch_dims=1)
            
            # ---------------マッチングスコア計算部分-----------------
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
            setMatchingloss = self.SetMatchingScore_loss(Real_score, Fake_score)
            # --------------------------------------------

            if not self.set_loss :
                if self.gallerytype == 'InBatch':
                    cos_sim_forloss = self.cos_similarity_pos_neg((gallery, predSMN), gallerytype=self.gallerytype)  
                    cos_sim_forrank = self.cos_similarity_pos_neg((pos_neg_G, predSMN), gallerytype='OutBatch') 
                else:
                    cos_sim_forloss = self.cos_similarity_pos_neg((pos_neg_G, predSMN), gallerytype=self.gallerytype) 
                    cos_sim_forrank = cos_sim_forloss
            elif self.calc_set_sim == 'CS':
                set_score = self.cross_set_score((gallery, predSMN), x_size)
            elif self.calc_set_sim == 'BERTscore':
                set_score = self.BERT_set_score((gallery, predSMN), x_size)
            else:
                print("指定された集合間類似度を測る関数は存在しません")
                sys.exit()
                
            # style loss related
            if self.style_method == 'item_style':
                # style_loss = self.style_content_loss(predSMN, gallery, pred_size)
                style_loss = self.contrastive_style_loss(predSMN, pos_neg_G, pred_size)
            elif self.style_method == 'DFA_style':
                # PIFR function - PIFR accuracy
                weights = self.compute_representation_weights(predSMN, gallery, x_size)
                # y_predごとにgalleryの各集合をPIFR weightで再構成
                weights_gallery = [tf.matmul(weights[i][j], gallery[j]) for i in range(predSMN.shape[0]) for j in range(gallery.shape[0])] 
                weights_gallery = tf.reshape(weights_gallery, [predSMN.shape[0], gallery.shape[0], gallery.shape[1], gallery.shape[2]])
                style_loss = self.style_content_loss(predSMN, weights_gallery, pred_size)
            else:
                style_loss = 0
            
            if self.is_L2_norm_loss:
                # l2_loss = self.L2_norm_loss(pred=predSMN, y=gallery, pred_size=pred_size)
                l2_loss = self.L2_norm_loss(pred=predSMN, y=before_pred, pred_size=pred_size)
            else:
                l2_loss = 0
            if not self.set_loss:
                loss = self.compiled_loss(y_pred = cos_sim_forloss, y_true = c_label, regularization_losses=self.losses) + style_loss * (1/20) + setMatchingloss + l2_loss
                
                # loss = self.compiled_loss(y_pred = cos_sim, y_true = c_label, regularization_losses=self.losses)
            else:
                loss = self.compiled_loss(set_score, y_true, regularization_losses=self.losses)
        # train using gradients
        trainable_vars = self.trainable_variables
        
        
        # train parameters excepts for CNN
        trainable_vars = [v for v in trainable_vars if 'cnn' not in v.name]
        trainable_vars = [v for v in trainable_vars if 'set_matching_model' not in v.name]

        gradients = tape.gradient(loss, trainable_vars)

        # for idx, grad in enumerate(gradients):
        #     if grad is None:
        #         print(f"Layer {idx}: Gradient is None!")
        #     else:
        #         print(f"Layer {idx}: Gradient mean={tf.reduce_mean(tf.abs(grad))}")
        
        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, trainable_vars)
            if grad is not None)

        # update metrics

        if not self.set_loss:
            self.compiled_metrics.update_state(cos_sim_forrank, pred_size)
            # return metrics as dictionary
            return {'cos_sim_loss': loss-style_loss*(1/20)-setMatchingloss-l2_loss, 'Match_loss': setMatchingloss, 'L2_loss': l2_loss,'Set_accuracy':self.metrics[1].result()}
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
            x, x_size, c_label, negative = x 
        #cross set label creation
        # y_true : [(1,0...,0),(0,1,...,0),...,(0,0,...,1)] locates where the positive is. (nSet, nSet)
        y_true = self.cross_set_label(y_true)
        y_true = tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))

        if not self.set_loss:
            # c_label (before): odd and even number index hasn't been shuffuled => c_label(after): each index indicates the answer
            ans_c_label = tf.gather(c_label, tf.where(tf.equal(y_true,1))[:,1])
            pred_size = tf.gather(x_size, tf.where(tf.equal(y_true,1))[:,1])
        
        # whitening_step
        if self.linear_parampath != None:
            x = np.matmul(x, self.gause_noise) # equals to fc(x) 
            # x = (x- self.whitening_mean) / self.whitening_std # normalization
        else:
            x = self.fc_cnn_proj(x)
        # gallery : (nSet, nItemMax, dim)
        gallery = x

        # negative gallery + positive gallery generating
        pos_G = tf.gather(gallery,tf.where(tf.equal(y_true,1))[:,1])
        pos_G = tf.expand_dims(pos_G, axis=2)
        negative = np.matmul(negative, self.gause_noise)
        # neg_G = tf.gather(negative, ans_c_label)
        pos_neg_G = tf.concat([pos_G, negative],axis=2)

        # predict
        # predSMN : (nSet, nItemMax, d)
        # y_pred = tf.tile(self.set_emb, [x.shape[0], 1,1]) # random init ver
        y_pred = tf.tile(tf.expand_dims(tf.stack(self.set_emb), axis=0),[x.shape[0],1,1])
        y_pred = tf.matmul(y_pred, tf.cast(self.gause_noise, tf.float32))
        y_pred = tf.gather(y_pred, ans_c_label, batch_dims=1)
        y_size = tf.constant(np.full(x.shape[0],41).astype(np.float32))
        if not self.set_loss:
            before_pred, predSMN, debug = self(((x, x_size), (y_pred, pred_size)), training=False)
        else:
            before_pred, predSMN, debug = self((x, x_size), training=False)
        

        # ---------------マッチングスコア計算部分------------------
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
        setMatchingloss = self.SetMatchingScore_loss(Real_score, Fake_score)
        # --------------------------------------------
        
        # --------------------------------------------
        if not self.set_loss :
            if self.gallerytype == 'InBatch':
                    cos_sim_forloss = self.cos_similarity_pos_neg((gallery, predSMN), gallerytype=self.gallerytype)  
                    cos_sim_forrank = self.cos_similarity_pos_neg((pos_neg_G, predSMN), gallerytype='OutBatch') 
            else:
                cos_sim_forloss = self.cos_similarity_pos_neg((pos_neg_G, predSMN), gallerytype=self.gallerytype) 
                cos_sim_forrank = cos_sim_forloss
        elif self.calc_set_sim == 'CS':
            set_score = self.cross_set_score((gallery, predSMN), x_size)
        elif self.calc_set_sim == 'BERTscore':
            set_score = self.BERT_set_score((gallery, predSMN), x_size)
        else:
            print("指定された集合間類似度を測る関数は存在しません")
            sys.exit()
            
        # style loss related
        if self.style_method == 'item_style':
            # style_loss = self.style_content_loss(predSMN, gallery, pred_size)
            style_loss = self.contrastive_style_loss(predSMN, pos_neg_G, pred_size)
        elif self.style_method == 'DFA_style': 
            # PIFR function - PIFR accuracy
            weights = self.compute_representation_weights(predSMN, gallery, x_size)
            # y_predごとにgalleryの各集合をPIFR weightで再構成
            weights_gallery = [tf.matmul(weights[i][j], gallery[j]) for i in range(predSMN.shape[0]) for j in range(gallery.shape[0])] 
            weights_gallery = tf.reshape(weights_gallery, [predSMN.shape[0], gallery.shape[0], gallery.shape[1], gallery.shape[2]])
            style_loss = self.style_content_loss(predSMN, weights_gallery, pred_size)
        else:
            style_loss = 0
        
        if self.is_L2_norm_loss:
            # l2_loss = self.L2_norm_loss(pred=predSMN, y=gallery, pred_size=pred_size)
            l2_loss = self.L2_norm_loss(pred=predSMN, y=before_pred, pred_size=pred_size)
        else:
            l2_loss = 0

        if not self.set_loss:
            loss = self.compiled_loss(y_pred = cos_sim_forloss, y_true = c_label, regularization_losses=self.losses) + style_loss * (1/20) + setMatchingloss + l2_loss
            # loss = self.compiled_loss(y_pred = cos_sim, y_true = c_label, regularization_losses=self.losses)
        else:
            loss = self.compiled_loss(set_score, y_true, regularization_losses=self.losses)

        if not self.set_loss:
            self.compiled_metrics.update_state(cos_sim_forrank, pred_size)
            # return metrics as dictionary
            return {'cos_sim_loss': loss-style_loss*(1/20)-setMatchingloss-l2_loss, 'Match_loss': setMatchingloss, 'L2_loss': l2_loss,'Set_accuracy':self.metrics[1].result()}
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

        if not self.set_loss:
            # c_label (before): odd and even number index hasn't been shuffuled => c_label(after): each index indicates the answer
            ans_c_label = tf.gather(c_label, tf.where(tf.equal(y_true,1))[:,1])
            pred_size = tf.gather(x_size, tf.where(tf.equal(y_true,1))[:,1])
        
        # whitening_step
        if self.linear_parampath != None:
            x = np.matmul(x, self.gause_noise) # equals to fc(x) 
            # x = (x- self.whitening_mean) / self.whitening_std # normalization
        else:
            x = self.fc_cnn_proj(x)
        # gallery : (nSet, nItemMax, dim)
        gallery = x
        # y_pred = tf.tile(self.set_emb, [x.shape[0], 1,1]) # random ver == no gause
        y_pred = tf.tile(tf.expand_dims(tf.stack(self.set_emb), axis=0),[x.shape[0],1,1])
        y_pred = tf.matmul(y_pred, tf.cast(self.gause_noise, tf.float32))
        y_pred = tf.gather(y_pred, ans_c_label, batch_dims=1)
        y_size = tf.constant(np.full(x.shape[0],41).astype(np.float32))
        # predict
        # predSMN : (nSet, nItemMax, d)
        # predCNN, predSMN, debug = self((x, x_size, [], ans_c_label, pred_size), training=False)
        before_pred, predSMN, debug = self(((x, x_size), (y_pred, y_size)), training=False)

        
        if len(x) <= 50:
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
        if len(x) <= 50:
            return predSMN, gallery, replicated_set_label, query_id, Real_score, Fake_score
        else:
            return predSMN, gallery, replicated_set_label, query_id
#----------------------------
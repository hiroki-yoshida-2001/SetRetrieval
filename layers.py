import tensorflow as tf
import tensorflow_addons as tfa
import os
import matplotlib.pylab  as plt
import numpy as np
import pdb
import scipy



#----------------------------------------------------------------------------------
# Layer Normalization
class layer_normalization(tf.keras.layers.Layer):
    def __init__(self, size_d, epsilon=1e-3, is_cross=False, is_instance_norm=False):
        super(layer_normalization, self).__init__()
        # self.gain = self.add_weight(name='gain', shape=(size_d), trainable=True, initializer=tf.ones_initializer())
        # self.bias = self.add_weight(name='bias', shape=(size_d), trainable=True, initializer=tf.zeros_initializer())
        self.epsilon = epsilon
        self.is_cross = is_cross
        self.is_instance_norm = is_instance_norm

    def call(self, x, x_size, y, y_size):
        
        x_nSet = x_size.shape[0]
        y_nSet = y_size.shape[0]

        if not self.is_instance_norm:
            if self.is_cross:
                x = tf.concat([x,tf.transpose(y,[1,0,2,3])], axis=2)
                x_size_tile = tf.tile(tf.expand_dims(x_size,1),[1,y_nSet])
                x_size_tile = x_size_tile + tf.expand_dims(y_size,0)
            else:
                shape = tf.shape(x)
                x_size_tile=tf.tile(tf.expand_dims(x_size,1),[1,shape[1]])
            # change shape
            shape = tf.shape(x)
            x_reshape = tf.reshape(x,[shape[0],shape[1],-1])

            # calc norm inside set
            mask = tf.cast(tf.not_equal(x_reshape,0),float)  
            if len(x_size_tile) == 16:
                pdb.set_trace()      
            mean_set = tf.reduce_sum(x_reshape,-1)/(x_size_tile*tf.cast(shape[-1],float))
            diff = x_reshape-tf.tile(tf.expand_dims(mean_set,-1),[1,1,shape[2]*shape[3]])
            std_set = tf.sqrt(tf.reduce_sum(tf.square(diff)*mask,-1)/(x_size_tile*tf.cast(shape[-1],float)))
        
            # output
            output = diff/tf.tile(tf.expand_dims(std_set,-1),[1,1,shape[2]*shape[3]])*mask
            output = tf.reshape(output,[shape[0],shape[1],shape[2],shape[3]])

            if self.is_cross:
                output = tf.split(output,2,axis=2)[0]
        else:
            # calc norm to D
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            std = tf.math.reduce_std(x, axis=-1, keepdims=True)
            norm = tf.divide((x - mean), std + self.epsilon)
            
            # create mask
            mask = tf.not_equal(x,0)

            # output
            output = tf.where(mask, norm, tf.zeros_like(x))

        return output



#----------------------------------------------------------------------------------
# multi-head CS function to make cros-set matching score map
class cross_set_score(tf.keras.layers.Layer):
    def __init__(self, head_size=20, num_heads=2):
        super(cross_set_score, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        #self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,kernel_constraint=tf.keras.constraints.NonNeg(),use_bias=False)
        self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,use_bias=False)
        self.linear2 = tf.keras.layers.Dense(1,use_bias=False)

    def call(self, x, x_size, y, y_size):
        nSet_x = tf.shape(x)[0]
        nSet_y = tf.shape(x)[1]
        nItemMax_x = tf.shape(x)[2]
        nItemMax_y = tf.shape(y)[2]
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))
        
        # linear transofrmation from (nSet_x, nSet_y, nItemMax, Xdim) to (nSet_x, nSet_y, nItemMax, head_size*num_heads)
        x = self.linear(x)
        y = self.linear(y)

        # reshape (nSet_x, nSet_y, nItemMax, head_size*num_heads) to (nSet_x, nSet_y, nItemMax, num_heads, head_size)
        # transpose (nSet_x, nSet_y, nItemMax, num_heads, head_size) to (nSet_x, nSet_y, num_heads, nItemMax, head_size)
        x = tf.transpose(tf.reshape(x,[nSet_x, nSet_y, nItemMax_x, self.num_heads, self.head_size]),[0,1,3,2,4])
        y = tf.transpose(tf.reshape(y,[nSet_y, nSet_x, nItemMax_y, self.num_heads, self.head_size]),[0,1,3,2,4])
        
        # compute inner products between all pairs of items with cross-set feature (cseft)
        # Between set #1 and set #2, cseft x[0,1] and x[1,0] are extracted to compute inner product when nItemMax=2
        # More generally, between set #i and set #j, cseft x[i,j] and x[j,i] are extracted.
        # Outputing (nSet_x, nSet_y, num_heads)-score map
        
        scores = tf.stack(
            [[
                tf.reduce_sum(tf.reduce_sum(
                tf.keras.layers.ReLU()(tf.matmul(x[i,j],tf.transpose(y[j,i],[0,2,1]))/sqrt_head_size)
                ,axis=1),axis=1)/x_size[i]/y_size[j]
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
            
        # linearly combine multi-head score maps (nSet_x, nSet_y, num_heads) to (nSet_x, nSet_y, 1)
        scores = self.linear2(scores)

        return scores



#----------------------------------------------------------------------------------
# cross-set feature (cseft)
class set_attention(tf.keras.layers.Layer):
    def __init__(self, head_size=20, num_heads=2, activation="softmax", self_attention=False):
        super(set_attention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads        
        self.activation = activation
        self.self_attention = self_attention
        self.cseft_with_self = False
        self.cseft_with_replace = False
        self.without_rep = False
        self.cseft_with_self_without_rep = False
        self.cseft_without_rep = False
        self.cross_modal_pivot_cross = False
        self.rep_vec_num = 1
        
        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        self.linear1 = tf.keras.layers.Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_attention')
        self.linear2 = tf.keras.layers.Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_attention')
        self.linear3 = tf.keras.layers.Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_attention')
        self.linear4 = tf.keras.layers.Dense(units=self.head_size, use_bias=False, name='set_attention')
        self.linear_xy = tf.keras.layers.Dense(units=self.head_size, use_bias=False, name='set_attention')
        #self.linear_yx = tf.keras.layers.Dense(units=self.head_size, use_bias=False, name='set_attention')


    def call(self, x, y):
        # number of sets
        nSet_x = tf.shape(x)[0]
        nSet_y = tf.shape(y)[0]
        nItemMax_x = tf.shape(x)[2]
        nItemMax_y = tf.shape(y)[2]
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))
        

        if self.self_attention:
            
            # input (nSet, nSet, nItemMax, dim)
            # linear transofrmation (nSet, nSet, nItemMax, head_size*num_heads)
            x = self.linear1(x)     # Query
            y1 = self.linear2(y)    # Key
            y2 = self.linear3(y)    # Value

            x = tf.reshape(x,[-1, nItemMax_x, self.head_size*self.num_heads])
            y1 = tf.reshape(y1,[-1, nItemMax_y, self.head_size*self.num_heads])
            y2 = tf.reshape(y2,[-1, nItemMax_y, self.head_size*self.num_heads])

            if self.without_rep:
                y1 = y1[:,self.rep_vec_num:]
                y2 = y2[:,self.rep_vec_num:]
                nItemMax_y -= self.rep_vec_num

        else:
            # input (nSet, nSet, nItemMax, dim)
            # transpose and reshape (nSet*nSet, nItemMax, dim)
            x = tf.reshape(tf.transpose(x,[1,0,2,3]),[-1, nItemMax_x, tf.shape(x)[-1]])   # nSet*nSet: (x1, x2, ..., x10, x1, x2, ..., x10, ...)
            y = tf.reshape(y,[-1, nItemMax_y, tf.shape(y)[-1]])   # nSet*nSet: (y1, y1, y1, ..., y2, y2, y2, ..., y8, y8, ...)

            # linear transofrmation (nSet, nSet, nItemMax, head_size*num_heads)
            if self.cross_modal_pivot_cross:
                # linear x space to y spaces
                x2 = self.linear_xy(x)
                #--------------------------------
                # concat y and linear x
                if self.cseft_with_self_without_rep: # single-PMA + pivot-cross
                    y = tf.concat([y, x2[:,self.rep_vec_num:]], axis=1)
                    nItemMax_y += nItemMax_x - self.rep_vec_num

                if self.cseft_without_rep: # single-PMA + pivot-cross
                    y = tf.concat([y[:,self.rep_vec_num:], x2[:,self.rep_vec_num:]], axis=1)
                    nItemMax_y += nItemMax_x - 2*self.rep_vec_num

                if self.cseft_with_self: # pivot-cross
                    y = tf.concat([y, x2],axis=1)
                    nItemMax_y += nItemMax_x

                if self.cseft_with_replace: # class-embedding + pivot-cross
                    y = tf.concat([x2[:,:self.rep_vec_num], y[:,self.rep_vec_num:]],axis=1)
                #--------------------------------
                # linear
                x = self.linear1(x)     # Query
                y1 = self.linear2(y)    # Key
                y2 = self.linear3(y)    # Values

            else:
                # linear
                x = self.linear1(x)     # Query
                y1 = self.linear2(y)    # Key
                y2 = self.linear3(y)    # Value

                #--------------------------------
                if self.without_rep:
                    y1 = y1[:,self.rep_vec_num:]
                    y2 = y2[:,self.rep_vec_num:]
                    nItemMax_y -= self.rep_vec_num

                if self.cseft_with_self_without_rep: # single-PMA + pivot-cross
                    y1 = tf.concat([y1, x[:,self.rep_vec_num:]],axis=1)
                    y2 = tf.concat([y2, x[:,self.rep_vec_num:]],axis=1)
                    nItemMax_y += nItemMax_x - self.rep_vec_num

                if self.cseft_without_rep: # single-PMA + pivot-cross
                    y1 = tf.concat([y1[:,self.rep_vec_num:], x[:,self.rep_vec_num:]],axis=1)
                    y2 = tf.concat([y2[:,self.rep_vec_num:], x[:,self.rep_vec_num:]],axis=1)
                    nItemMax_y += nItemMax_x - 2*self.rep_vec_num

                if self.cseft_with_self: # pivot-cross
                    y1 = tf.concat([y1, x],axis=1)
                    y2 = tf.concat([y2, x],axis=1)
                    nItemMax_y += nItemMax_x

                if self.cseft_with_replace: # class-embedding + pivot-cross
                    y1 = tf.concat([x[:,:self.rep_vec_num], y1[:,self.rep_vec_num:]],axis=1)
                    y2 = tf.concat([x[:,:self.rep_vec_num], y2[:,self.rep_vec_num:]],axis=1)
                #--------------------------------

        # # reshape (nSet*nSet, nItemMax, num_heads*head_size) to (nSet*nSet, nItemMax, num_heads, head_size)
        # # transpose (nSet*nSet, nItemMax, num_heads, head_size) to (nSet*nSet, num_heads, nItemMax, head_size)
        x = tf.transpose(tf.reshape(x,[-1, nItemMax_x, self.num_heads, self.head_size]),[0,2,1,3])
        y1 = tf.transpose(tf.reshape(y1,[-1, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])
        y2 = tf.transpose(tf.reshape(y2,[-1, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])

        # inner products between all pairs of items, outputing (nSet*nSet, num_heads, nItemMax_x, nItemMax_y)-score map    
        xy1 = tf.matmul(x,tf.transpose(y1,[0,1,3,2]))/sqrt_head_size

        def masked_softmax(x):
            # 0 value is treated as mask
            mask = tf.not_equal(x,0)
            x_exp = tf.where(mask,tf.exp(x-tf.reduce_max(x,axis=-1,keepdims=1)),tf.zeros_like(x))
            softmax = x_exp/(tf.reduce_sum(x_exp,axis=-1,keepdims=1) + 1e-10)
            return softmax

        # normalized by softmax
        attention_weight = masked_softmax(xy1)

        # computing weighted y2, outputing (nSet*nSet, num_heads, nItemMax_x, head_size)
        weighted_y2s = tf.matmul(attention_weight, y2)

        # reshape (nSet*nSet, num_heads, nItemMax_x, head_size) to (nSet*nSet, nItemMax_x, head_size*num_heads)
        weighted_y2s = tf.reshape(tf.transpose(weighted_y2s,[0,2,1,3]),[-1, nItemMax_x, self.num_heads*self.head_size])
        
        # combine multi-head to (nSet*nSet, nItemMax_x, head_size)
        output = self.linear4(weighted_y2s)

        if not self.self_attention:
            output = tf.transpose(tf.reshape(output,[nSet_y, nSet_x, nItemMax_x, self.head_size]),[1,0,2,3])
            #output = tf.reshape(output,[nSet_x, nSet_y, nItemMax_x, self.head_size])

        else:
            output = tf.reshape(output,[nSet_x, nSet_y, nItemMax_x, self.head_size])

        return output
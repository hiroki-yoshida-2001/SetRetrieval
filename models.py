import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import pdb
import sys

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
            x_reshape = tf.reshape(x,[shape[0],-1])

            # zero-padding mask
            mask = tf.reshape(tf.tile(tf.cast(tf.reduce_sum(x,axis=-1,keepdims=1)!=0,float),[1,1,shape[-1]]),[shape[0],-1])
            # mask = tf.cast(tf.not_equal(x_reshape,0),float)  
            # mean and std of set
            mean_set = tf.reduce_sum(x_reshape,-1)/(x_size*tf.cast(shape[-1],float))
            diff = x_reshape-tf.tile(tf.expand_dims(mean_set,-1),[1,shape[1]*shape[2]])
            std_set = tf.sqrt(tf.reduce_sum(tf.square(diff)*mask,-1)/(x_size*tf.cast(shape[-1],float)))
        
            # output
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

#----------------------------
# multi-head CS function to make cros-set matching score map
class cross_set_score(tf.keras.layers.Layer):
    def __init__(self, head_size=20, num_heads=2):
        super(cross_set_score, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        # self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,kernel_constraint=tf.keras.constraints.NonNeg(),use_bias=False)
        self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,use_bias=False)
        self.linear2 = tf.keras.layers.Dense(1,use_bias=False)

    def call(self, x, y, nItem):
        nSet_x = tf.shape(x)[0]
        nSet_y = tf.shape(y)[0]
        nItemMax_x = tf.shape(x)[1]
        nItemMax_y = tf.shape(y)[1]

        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))
        
        # linear transofrmation from (nSet_x, nItemMax_x, dim) to (nSet_x, nItemMax_x, head_size*num_heads)
        # linear transofrmation from (nSet_y, nItemMax_y, dim) to (nSet_y, nItemMax_y, head_size*num_heads)

        lx = self.linear(x)
        ly = self.linear(y)

        # reshape (nSet_x, nItemMax_x, head_size*num_heads) to (nSet_x, nItemMax_x, num_heads, head_size)
        # transpose (nSet_x, nItemMax_x, num_heads, head_size) to (nSet_x, num_heads, nItemMax_x, head_size) , * ly is transposed in the same way
        lx = tf.transpose(tf.reshape(lx,[nSet_x, nItemMax_x, self.num_heads, self.head_size]),[0,2,1,3])
        ly = tf.transpose(tf.reshape(ly,[nSet_y, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])
        
        # compute cosine similarity  between all pairs of items
        # x : (nSet_x, nItemMax_x, dim), y : (nSet_y, nItemMax_y, dim)
        # x, y is normalized in the direction of dim 
        # Outputing (nSet_x, nSet_y, nItemMax_x, nItemMax_y) -> cosine similarity (between all pairs)
        # e.g, cos_sim[1][0] (nItemMax_x, nItemMax_y) means cosine similarity between x[1] (nItemMax_x, dim) and y[0] (nItemMax_y, dim)
     
        cos_sim = tf.stack(
            [[                
                tf.matmul(tf.nn.l2_normalize(x[i], axis=-1),tf.transpose(tf.nn.l2_normalize(y[j], axis=-1),[1,0]))
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
        
        # compute inner products between all pairs of items with cross-set feature (cseft)
        # lx : (nSet_x, num_heads, nItemMax_x, head_size), ly : (nSet_y, num_heads, nItemMax_y, head_size)
        # Outputing (nSet_x, nSet_y, head_size) -> Cross similarity 

        scores = tf.stack(
            [[
                tf.reduce_sum(tf.reduce_sum(
                tf.keras.layers.ReLU()(tf.matmul(lx[i],tf.transpose(ly[j],[0,2,1]))/sqrt_head_size)
                , axis=1), axis=1)/nItem[i]/nItem[j]
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
            
        # linearly combine multi-head score maps (nSet_x, nSet_y, num_heads) to (nSet_x, nSet_y, 1)
        # e.g, scores [1][0] (1,) means Cross similarity score between x[1] (nItemMax_x, dim) and y[0] (nItemMax_y, dim)
        scores = self.linear2(scores)

        return cos_sim, scores
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

#----------------------------
# set matching network
class SMN(tf.keras.Model):
    def __init__(self, isCNN=True, is_set_norm=False, is_cross_norm=True, is_final_linear=True, num_layers=1, num_heads=2, mode='setRepVec_pivot', calc_set_sim='BERTscore', baseChn=32, rep_vec_num=1, cnn_class_num=2, max_channel_ratio=2, is_neg_down_sample=False):
        super(SMN, self).__init__()
        self.isCNN = isCNN
        self.num_layers = num_layers
        self.mode = mode
        self.calc_set_sim = calc_set_sim
        self.rep_vec_num = rep_vec_num
        self.baseChn = baseChn
        self.is_final_linear = is_final_linear
        self.is_neg_down_sample = is_neg_down_sample

        #---------------------
        # cnn
        self.CNN = []
        self.fc_cnn_proj = tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching')
        #---------------------
        
        #---------------------
        # encoder for query X
        self.set_emb = self.add_weight(name='set_emb',shape=(1,self.rep_vec_num,baseChn*max_channel_ratio),trainable=True)
        self.self_attentionsX = [set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads, self_attention=True) for i in range(num_layers)]
        self.layer_norms_enc1X = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.layer_norms_enc2X = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.fcs_encX = [tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching') for i in range(num_layers)]        
        #---------------------
        # encoder for seed 
        self.self_attentionsR = [set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads, self_attention=True) for i in range(num_layers)]
        self.layer_norms_enc1R = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.layer_norms_enc2R = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.fcs_encR = [tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching') for i in range(num_layers)]  
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

    def call(self, x):
        x, x_size = x
        debug = {}
        shape = tf.shape(x)
        nSet = shape[0]
        nItemMax = shape[1]

        # CNN
        if self.isCNN:
            x, predCNN = self.CNN((x,x_size),training=False)
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
        # add_embedding
        y_seed = tf.tile(self.set_emb, [nSet,1,1]) # (nSet, nItemMax, D)
        y_seed_size = tf.constant(np.full(nSet,self.rep_vec_num).astype(np.float32))

        #---------------------
        # decoder (cross-attention)
        debug[f'x_decoder_layer_0'] = x
        for i in range(self.num_layers):
     
            if self.mode == 'setRepVec_pivot': # Bi-PMA + pivot-cross
                self.cross_attentions[i].pivot_cross = True

            query = self.layer_norms_decq[i](y_seed,y_seed_size)
            key = self.layer_norms_deck[i](x,x_size)

            # input: (nSet, nItemMax, D), output:(nSet, nItemMax, D)
            query = self.cross_attentions[i](query,key)
            y_seed += query
    
            query = self.layer_norms_dec2[i](y_seed,y_seed_size)
            

            query = self.fcs_dec[i](query)
            y_seed += query

            debug[f'x_decoder_layer_{i+1}'] = x
        x_dec = x
        #---------------------

        #---------------------
        
        # #---------------------

        return predCNN, y_seed, debug
    
    # compute cosine similarity between all pairs of items and BERTscore
    def BERT_set_score(self, x, y, beta=0.2):

        nSet_x, nItemMax_x, dim = x.shape 
        nSet_y, nItemMax_y, dimg = y.shape

        # compute cosine similarity  between all pairs of items
        # x : (nSet_x, nItemMax_x, dim), y : (nSet_y, nItemMax_y, dim)
        # x, y is normalized in the direction of dim 
        # Outputing (nSet_x, nSet_y, nItemMax_x, nItemMax_y) -> cosine similarity (between all pairs)
        # e.g, cos_sim[1][0] (nItemMax_x, nItemMax_y) means cosine similarity between x[1] (nItemMax_x, dim) and y[0] (nItemMax_y, dim)

        cos_sim = tf.stack(
            [[                
                tf.matmul(tf.nn.l2_normalize(x[i], axis=-1),tf.transpose(tf.nn.l2_normalize(y[j], axis=-1),[1,0]))
                for i in range(nSet_x)] for j in range(nSet_y)]
            )

        '''for batch_ind in range(len(cos_sim)):

            score = cos_sim[batch_ind] # (nSet_y, nItemMax_x, nItemMax_y)

            #-----------------------------------------------
            # score_for_recall: ギャラリの各要素から最近傍の予測集合の要素を選択, score_for_precision: 予測集合の各要素から最近傍のギャラリの要素を選択

            # softmax is applied in row direction (nItemMax_x)
            # average softmax score in row direction (nItemMax_x) => score_for_recall : (nSet_y, nItemMax_y)
            score_for_recall = tf.reduce_mean(tf.nn.softmax(score,axis=1), axis=1)

            # softmax is applied in column direction (nItemMax_y)
            # average softmax score in column direction (nItemMax_y) => score_for_precision : (nSet_y, nItemMax_x)
            score_for_precision = tf.reduce_mean(tf.nn.softmax(score,axis=2), axis=2)

            #-------------------------------------------------
            # 各要素の最近傍を平均して集合としての最近傍を表現

            # score_for_precision : (nSet_y, nItemMax_y) => precision_score : (nSet_y, 1)
            # score_for_recall : (nSet_y, nItemMax_x) => recall_score : (nSet_y, 1)
            precision_score = tf.reduce_mean(score_for_precision, axis=1, keepdims=True)
            recall_score = tf.reduce_mean(score_for_recall, axis=1, keepdims=True)
            f1_score = 2 * (precision_score*recall_score) / (precision_score+recall_score)
            # f1_score = (precision_score*recall_score*(1+beta**2))/(precision_score+ beta**2 *recall_score)
            #-------------------------------------------------

            if batch_ind == 0:
                f1_scores = tf.expand_dims(f1_score,axis=0)
            else:
                f1_scores = tf.concat([f1_scores, tf.expand_dims(f1_score,axis=0)], axis=0)'''

        # cos_sim[i] (x[i]に対する予測ベクトルとyの各アイテムとの類似度) : (nSet_y, nItemMax_x, nItemMax_y)

        # As to caluclating recall,  softmax is applied in row direction (nItemMax_x) -> tf.nn.softmax(cos_sim[i], axis=1) 
            # average softmax score (tf.nn.softmax(cos_sim[i], axis=1)) in row direction (nItemMax_x) => score_for_recall : (nSet_y, nItemMax_y)
                # average each item score  => tf.reduce_mean(tf.reduce_mean(tf.nn.softmax(cos_sim[i], axis=2), axis=2), axis=1, keepdims=True) : (nSet_y, 1)
        
        # As to caluclating precision,  softmax is applied in column direction (nItemMax_y) -> tf.nn.softmax(cos_sim[i], axis=2) 
            # average softmax score (tf.nn.softmax(cos_sim[i], axis=2)) in column direction (nItemMax_y) => score_for_recall : (nSet_y, nItemMax_x)
                # average each item score =>  tf.reduce_mean(tf.nn.softmax(cos_sim[i], axis=2), axis=2), axis=1, keepdims=True) : (nSet_y, 1)
        
        f1_scores = [
            
                2 * (
                    tf.reduce_mean(tf.reduce_mean(tf.nn.softmax(cos_sim[i], axis=1), axis=1), axis=1, keepdims=True) *
                    tf.reduce_mean(tf.reduce_mean(tf.nn.softmax(cos_sim[i], axis=2), axis=2), axis=1, keepdims=True)
                ) / (
                    tf.reduce_mean(tf.reduce_mean(tf.nn.softmax(cos_sim[i], axis=1), axis=1), axis=1, keepdims=True) +
                    tf.reduce_mean(tf.reduce_mean(tf.nn.softmax(cos_sim[i], axis=2), axis=2), axis=1, keepdims=True)
                )
            for i in range(len(cos_sim))
        ]
        f1_scores = tf.stack(f1_scores, axis=0)

        return cos_sim, f1_scores
    
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

    def neg_down_sampling(self, y_true, y_pred):
        # split to positive or negative data
        mask_pos = tf.not_equal(y_true,0)
        mask_neg = tf.not_equal(y_true,1)
        
        # number of pos and neg
        num_pos = tf.reduce_sum(tf.cast(mask_pos,tf.int32))
        num_neg = tf.reduce_sum(tf.cast(mask_neg,tf.int32))
        
        # split
        y_true_pos = tf.boolean_mask(y_true,mask_pos)
        y_pred_pos = tf.boolean_mask(y_pred,mask_pos)
        y_true_neg = tf.boolean_mask(y_true,mask_neg)
        y_pred_neg = tf.boolean_mask(y_pred,mask_neg)

        # select neg data
        # select neg data
        thre = tf.cast(1.0-num_pos/num_neg,float)
        mask_neg_thre = tf.greater(tf.random.uniform([num_neg]),thre)
        y_true_neg = tf.boolean_mask(y_true_neg,mask_neg_thre)
        y_pred_neg = tf.boolean_mask(y_pred_neg,mask_neg_thre)

        # concat
        y_true = tf.concat([y_true_pos,y_true_neg],axis=0)
        y_pred = tf.concat([y_pred_pos,y_pred_neg],axis=0)

        return y_true, y_pred
    
    def swap_query_positive(self, array):
        # クエリとポジティブの位置を設定するために、セットのインデックスを交換する関数
        indices = tf.range(0, tf.shape(array)[0])
        swapped_indices = tf.reshape(tf.stack([indices[1::2], indices[::2]], axis=-1), [-1])

        return swapped_indices
    # identify positive set position and return the size of the set
    def get_positive_set_item__num(self, x_size):
        array = x_size
        swapped_indices = self.swap_query_positive(array)
        positive_set_item_num = tf.gather(array, swapped_indices)

        return positive_set_item_num

    # train step
    def train_step(self,data):
        x, y_true = data
        x, x_size = x # xの順番は[a_1,a_2,b_1,b_2, c_1,c_2,...]

        # クエリとポジティブを含むすべてのアイテムのデータベース
        gallery = x 
        setlabel = y_true # 可視化の時用に集合ラベルを保存

        # gallery linear projection(dimmension reduction) 
        gallery = self.fc_cnn_proj(gallery)

        with tf.GradientTape() as tape:
            # predict
            predCNN, predSMN, debug = self((x, x_size), training=True)
            
            y_pred = predSMN

            # cross set label creation
            y_true = self.cross_set_label(y_true)
            y_true= tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))

            # ポジティブ集合の要素数の参照
            y_true_num = self.get_positive_set_item__num(x_size)

            # compute similairty with gallery and f1_bert_score
            if self.calc_set_sim == 'CS':
                similarity, set_score = self.cross_set_score(predSMN, gallery, y_true_num)
            elif self.calc_set_sim == 'BERTscore':
                similarity, set_score = self.BERT_set_score(predSMN, gallery)
            else:
                print("指定された集合間類似度を測る関数は存在しません")
                sys.exit()

            # # down sampling
            # if self.is_neg_down_sample:
            #     y_true, y_pred = self.neg_down_sampling(y_true, y_pred)

            # loss
            loss = self.compiled_loss(similarity, set_score, regularization_losses=self.losses)
     
        # train using gradients
        trainable_vars = self.trainable_variables

        # train parameters excepts for CNN
        trainable_vars = [v for v in trainable_vars if 'cnn' not in v.name]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, trainable_vars)
            if grad is not None)

        # update metrics
        self.compiled_metrics.update_state(y_true, set_score)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):
 
        x, y_true = data
        x , x_size = x

        # クエリとポジティブを含むすべてのアイテムのデータベース
        gallery = x 
        setlabel = y_true # 可視化の時用に集合ラベルを保存

        # gallery linear projection(dimmension reduction) 
        gallery = self.fc_cnn_proj(gallery)

        # predict
        predCNN, predSMN, debug = self((x, x_size), training=False)
        y_pred = predSMN

        #cross set label creation
        y_true = self.cross_set_label(y_true)
        y_true= tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))
        
        # ポジティブ集合の要素数の参照
        y_true_num = self.get_positive_set_item__num(x_size)

        # compute similairty with gallery and f1_bert_score
        if self.calc_set_sim == 'CS':
            similarity, set_score = self.cross_set_score(predSMN, gallery, y_true_num)
        elif self.calc_set_sim == 'BERTscore':
            similarity, set_score = self.BERT_set_score(predSMN, gallery)
        else:
            print("指定された集合間類似度を測る関数は存在しません")
            sys.exit()

        # # down sampling
        # if self.is_neg_down_sample:
        #     y_true, y_pred = self.neg_down_sampling(y_true, y_pred)

        # loss
        self.compiled_loss(similarity, set_score, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(y_true, set_score)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # predict step
    def predict_step(self,data):
        batch_data = data[0]
        x, x_size = batch_data
        gallery = x
        gallery = self.fc_cnn_proj(gallery)
        # predict
        predCNN, predSMN, debug = self((x, x_size), training=False)

        # compute similairty with gallery and f1_bert_score
        if self.calc_set_sim == 'CS':
            sys.exit()
        elif self.calc_set_sim == 'BERTscore':
            similarity, set_score = self.BERT_set_score(predSMN, gallery)
        else:
            print("指定された集合間類似度を測る関数は存在しません")
            sys.exit()

        return predSMN, similarity, set_score
#----------------------------


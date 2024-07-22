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

            scores = tf.stack(
            [[
                tf.reduce_sum(tf.reduce_sum(
                tf.keras.layers.LeakyReLU()(tf.matmul(y[j],tf.transpose(x[i],[0,2,1]))/sqrt_head_size)
                , axis=1), axis=1)/nItem[i]/tf.cast(nItemMax_y, tf.float32)
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
             
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
    def __init__(self, baseChn=1024, category_class_num=41):
        super(MLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(baseChn, activation=tfa.activations.gelu, use_bias=False)
        self.fc2 = tf.keras.layers.Dense(baseChn//2, activation=tfa.activations.gelu, use_bias=False)
        self.fc3 = tf.keras.layers.Dense(baseChn//4, activation=tfa.activations.gelu, use_bias=False)
        self.fc4 = tf.keras.layers.Dense(category_class_num, activation='softmax', use_bias=False)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
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

#----------------------------
# set matching network
class SMN(tf.keras.Model):
    def __init__(self, isCNN=True, is_set_norm=False, is_cross_norm=True, is_TrainableMLP=True, num_layers=1, num_heads=2, mode='setRepVec_pivot', calc_set_sim='BERTscore', baseChn=32, baseMlp = 512, rep_vec_num=1, seed_init = 0, cnn_class_num=2, max_channel_ratio=2, is_neg_down_sample=False, use_Cvec=True, is_Cvec_linear=False):
        super(SMN, self).__init__()
        self.isCNN = isCNN
        self.num_layers = num_layers
        self.mode = mode
        self.calc_set_sim = calc_set_sim
        self.rep_vec_num = rep_vec_num
        self.seed_init = seed_init
        self.baseChn = baseChn
        self.isTrainableMLP = is_TrainableMLP
        self.baseMlpChn = baseMlp
        self.is_neg_down_sample = is_neg_down_sample
        self.use_Cvec = use_Cvec
        self.is_Cvec_linear = is_Cvec_linear
        
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

        #---------------------
        # seed_vec initialization with cluster vectors
        if self.seed_init == 0:
            self.set_emb = self.add_weight(name='set_emb',shape=(1, self.rep_vec_num,baseChn*max_channel_ratio),trainable=True)
        else:
            # 4096 => 64次元への写像処理が必要
            self.set_emb = [self.add_weight(name='set_emb',shape=(self.dim_shift15,),initializer=self.custom_initializer(self.seed_init[i]),trainable=True) for i in range(len(self.seed_init))]
        #---------------------
        # MLP models
        self.fc1 = tf.keras.layers.Dense(baseMlp, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn')
        self.fc2 = tf.keras.layers.Dense(baseMlp//2, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn')
        self.fc3 = tf.keras.layers.Dense(baseMlp//4, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn')
        self.fc4 = tf.keras.layers.Dense(len(seed_init), activation='softmax', use_bias=False, name='setmatching_cnn')

    def custom_initializer(self, initial_values):
        def initializer(shape, dtype=None):
            # 次元ごとに異なる値を持つTensorを作成
            values = [initial_values[i] for i in range(shape[0])]
            return tf.constant(values, dtype=dtype)
        return initializer
    
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
            if self.isTrainableMLP:
                x = self.fc1(x)
                x = self.fc2(x)
                x = self.fc3(x)

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
        if self.use_Cvec:
            if self.isTrainableMLP:
                if self.seed_init == 0:
                    y_pred = tf.tile(self.set_emb, [nSet, 1,1])
                else:
                    y_pred = tf.tile(tf.expand_dims(tf.stack(self.set_emb), axis=0),[nSet,1,1])
                    y_pred = self.fc1(y_pred)
                    y_pred = self.fc2(y_pred)
                    y_pred = self.fc3(y_pred)
                    
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
            y_pred_size = tf.constant(np.full(nSet,self.rep_vec_num).astype(np.float32))

            #---------------------
            # decoder (cross-attention)
            debug[f'x_decoder_layer_0'] = x
            for i in range(self.num_layers):
        
                if self.mode == 'setRepVec_pivot': # Bi-PMA + pivot-cross
                    self.cross_attentions[i].pivot_cross = True

                query = self.layer_norms_decq[i](y_pred,y_pred_size)
                key = self.layer_norms_deck[i](x,x_size)

                # input: (nSet, nItemMax, D), output:(nSet, nItemMax, D)
                query = self.cross_attentions[i](query,key)
                y_pred += query
        
                query = self.layer_norms_dec2[i](y_pred,y_pred_size)
                

                query = self.fcs_dec[i](query)
                y_pred += query

                debug[f'x_decoder_layer_{i+1}'] = x
            x_dec = x
            #---------------------
        else: # text generation methods (only query X )
            y_pred = x
        #---------------------
        

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

            cos_sim = tf.stack(
            [[                
                tf.matmul(tf.nn.l2_normalize(y[j], axis=-1),tf.transpose(tf.nn.l2_normalize(x[i], axis=-1),[1,0]))
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
            f1_scores = [
            
                2 * (
                    tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=1), axis=1, keepdims=True) / tf.expand_dims(nItem,axis=1) *
                    tf.reduce_sum(tf.reduce_max(tf.where(tf.not_equal(cos_sim[i], 0), cos_sim[i], tf.fill(cos_sim[i].shape, float('-inf'))), axis=2), axis=1, keepdims=True) / tf.cast(nItemMax_y, tf.float32)
                ) / (
                    tf.reduce_sum(tf.reduce_max(cos_sim[i], axis=1), axis=1, keepdims=True) / tf.expand_dims(nItem,axis=1) +
                    tf.reduce_sum(tf.reduce_max(tf.where(tf.not_equal(cos_sim[i], 0), cos_sim[i], tf.fill(cos_sim[i].shape, float('-inf'))), axis=2), axis=1, keepdims=True) / tf.cast(nItemMax_y, tf.float32)
                )
                for i in range(len(cos_sim))
            ]
        
        # ------------------------------
        f1_scores = tf.stack(f1_scores, axis=0)

        return f1_scores
    
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
    # train step
    def train_step(self,data):
    
        # x = {x, x_size}, y_true : set label to identify positive pair. (nSet, )
        x, y_true = data
        # x : (nSet, nItemMax, dim) , x_size : (nSet, )
        x, x_size = x 
        
        # gallery : (nSet, nItemMax, dim)
        gallery = x
        # gallery linear projection(dimmension reduction) 
        if self.isTrainableMLP:
            gallery = self.fc1(gallery)
            gallery = self.fc2(gallery)
            gallery = self.fc3(gallery)
        else:
            gallery = self.fc_cnn_proj(gallery) # : (nSet, nItemMax, d=baseChn*max_channel_ratio)

        with tf.GradientTape() as tape:
            # predict
            # predSMN : (nSet, nItemMax, d)
            predCNN, predSMN, debug = self((x, x_size), training=True)
            
            # cross set label creation
            # y_true : [(1,0...,0),(0,1,...,0),...,(0,0,...,1)] locates where the positive is. (nSet, nSet) 
            y_true = self.cross_set_label(y_true)
            y_true = tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))


            # compute similairty with gallery and f1_bert_score
            # input gallery as x and predSMN as y in each bellow set similarity function. 
            if self.calc_set_sim == 'CS':
                set_score = self.cross_set_score((gallery, predSMN), x_size)
            elif self.calc_set_sim == 'BERTscore':
                set_score = self.BERT_set_score((gallery, predSMN), x_size)
            else:
                print("指定された集合間類似度を測る関数は存在しません")
                sys.exit()

            # # down sampling
            # if self.is_neg_down_sample:
            #     y_true, y_pred = self.neg_down_sampling(y_true, y_pred)

            # loss
            loss = self.compiled_loss(set_score, y_true, regularization_losses=self.losses)

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
        self.compiled_metrics.update_state(set_score, y_true)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):

        # x = {x, x_size}, y_true : set label to identify positive pair. (nSet, )
        x, y_true = data
        # x : (nSet, nItemMax, dim) , x_size : (nSet, )
        x , x_size = x
        # gallery : (nSet, nItemMax, dim)
        gallery = x 

        # gallery linear projection(dimmension reduction) 
        if self.isTrainableMLP:
            gallery = self.fc1(gallery)
            gallery = self.fc2(gallery)
            gallery = self.fc3(gallery)
        else:
            gallery = self.fc_cnn_proj(gallery) # : (nSet, nItemMax, d=baseChn*max_channel_ratio)

        # predict
        # predSMN : (nSet, nItemMax, d)
        predCNN, predSMN, debug = self((x, x_size), training=False)
        
        #cross set label creation
        # y_true : [(1,0...,0),(0,1,...,0),...,(0,0,...,1)] locates where the positive is. (nSet, nSet) 
        y_true = self.cross_set_label(y_true)
        y_true = tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))

        # compute similairty with gallery and f1_bert_score
        # input gallery as x and predSMN as y in each bellow set similarity function. 
        if self.calc_set_sim == 'CS':
            set_score = self.cross_set_score((gallery, predSMN), x_size)
        elif self.calc_set_sim == 'BERTscore':
            set_score = self.BERT_set_score((gallery, predSMN), x_size)
        else:
            print("指定された集合間類似度を測る関数は存在しません")
            sys.exit()

        # # down sampling
        # if self.is_neg_down_sample:
        #     y_true, y_pred = self.neg_down_sampling(y_true, y_pred)

        # loss
        self.compiled_loss(set_score, y_true, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(set_score, y_true)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # predict step
    def predict_step(self,data):
        
        batch_data = data[0]
        # x = {x, x_size}, y_true : set label to identify positive pair. (nSet, )
        x, x_size, y_test, item_label  = batch_data
        # gallery : (nSet, nItemMax, dim)
        gallery = x
        # gallery linear projection(dimmension reduction) 
        if self.isTrainableMLP:
            gallery = self.fc1(gallery)
            gallery = self.fc2(gallery)
            gallery = self.fc3(gallery)
        else:
            gallery = self.fc_cnn_proj(gallery) # : (nSet, nItemMax, d=baseChn*max_channel_ratio)

        # predict
        # predSMN : (nSet, nItemMax, d)
        predCNN, predSMN, debug = self((x, x_size), training=False)

        set_label = tf.cast(y_test, tf.int64)
        replicated_set_label = tf.tile(tf.expand_dims(set_label, axis=1), [1, len(x[0])])
        query_id = tf.stack([replicated_set_label, item_label],axis=1)
        query_id = tf.transpose(query_id, [0,2,1])

        return predSMN, gallery, replicated_set_label, query_id
#----------------------------
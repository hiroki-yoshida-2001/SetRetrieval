import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import pdb

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
        #self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,kernel_constraint=tf.keras.constraints.NonNeg(),use_bias=False)
        self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,use_bias=False)
        self.linear2 = tf.keras.layers.Dense(1,use_bias=False)

    def call(self, x, nItem):
        nSet_x = tf.shape(x)[0]
        nSet_y = tf.shape(x)[1]
        nItemMax = tf.shape(x)[2]
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))
        
        # linear transofrmation from (nSet_x, nSet_y, nItemMax, Xdim) to (nSet_x, nSet_y, nItemMax, head_size*num_heads)
        x = self.linear(x)

        # reshape (nSet_x, nSet_y, nItemMax, head_size*num_heads) to (nSet_x, nSet_y, nItemMax, num_heads, head_size)
        # transpose (nSet_x, nSet_y, nItemMax, num_heads, head_size) to (nSet_x, nSet_y, num_heads, nItemMax, head_size)
        x = tf.transpose(tf.reshape(x,[nSet_x, nSet_y, nItemMax, self.num_heads, self.head_size]),[0,1,3,2,4])
        
        # compute inner products between all pairs of items with cross-set feature (cseft)
        # Between set #1 and set #2, cseft x[0,1] and x[1,0] are extracted to compute inner product when nItemMax=2
        # More generally, between set #i and set #j, cseft x[i,j] and x[j,i] are extracted.
        # Outputing (nSet_x, nSet_y, num_heads)-score map        
        scores = tf.stack(
            [[
                tf.reduce_sum(tf.reduce_sum(
                tf.keras.layers.ReLU()(tf.matmul(x[j,i],tf.transpose(x[i,j],[0,2,1]))/sqrt_head_size)
                ,axis=1),axis=1)/nItem[i]/nItem[j]
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
            
        # linearly combine multi-head score maps (nSet_x, nSet_y, num_heads) to (nSet_x, nSet_y, 1)
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

        if self.self_attention:
            x = tf.reshape(x,[-1, nItemMax_x, self.head_size])
            y = tf.reshape(y,[-1, nItemMax_y, self.head_size])

        else:   # cross-attention
            pass
            # x = tf.reshape(tf.transpose(x,[1,0,2]),[-1, nItemMax_x, self.head_size])   # nSet*nSet: (x1, x2, x3, ..., x10, x1, x2, x3, ..., x10, ...)
            # y = tf.reshape(y,[-1, nItemMax_y, self.head_size])   # nSet*nSet: (y1, y1, y1, ..., y2, y2, y2, ..., y10, y10, y10)      

        # input (nSet, nSet, nItemMax, dim)
        # linear transofrmation (nSet, nSet, nItemMax, head_size*num_heads)
        y_K = self.linearK(y)   # Key
        y_V = self.linearV(y)   # Value
        x = self.linearQ(x)     # Query

        if self.pivot_cross: # pivot-cross
            y_K = tf.concat([y_K, x],axis=1)   # Key
            y_V = tf.concat([y_V, x],axis=1)   # Value            
            nItemMax_y += nItemMax_x

        # reshape (nSet*nSet, nItemMax, num_heads*head_size) to (nSet*nSet, nItemMax, num_heads, head_size)
        # transpose (nSet*nSet, nItemMax, num_heads, head_size) to (nSet*nSet, num_heads, nItemMax, head_size)
        x = tf.transpose(tf.reshape(x,[-1, nItemMax_x, self.num_heads, self.head_size]),[0,2,1,3])
        y_K = tf.transpose(tf.reshape(y_K,[-1, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])
        y_V = tf.transpose(tf.reshape(y_V,[-1, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])

        # inner products between all pairs of items, outputing (nSet*nSet, num_heads, nItemMax_x, nItemMax_y)-score map    
        xy_K = tf.matmul(x,tf.transpose(y_K,[0,1,3,2]))/sqrt_head_size

        def masked_softmax(x):
            # 0 value is treated as mask
            mask = tf.not_equal(x,0)
            x_exp = tf.where(mask,tf.exp(x-tf.reduce_max(x,axis=-1,keepdims=1)),tf.zeros_like(x))
            softmax = x_exp/(tf.reduce_sum(x_exp,axis=-1,keepdims=1) + 1e-10)

            return softmax

        # normalized by softmax
        attention_weight = masked_softmax(xy_K)
        # computing weighted y_V, outputing (nSet*nSet, num_heads, nItemMax_x, head_size)
        weighted_y_Vs = tf.matmul(attention_weight, y_V)

        # reshape (nSet*nSet, num_heads, nItemMax_x, head_size) to (nSet*nSet*nItemMax_x, head_size*num_heads)
        weighted_y_Vs = tf.reshape(tf.transpose(weighted_y_Vs,[0,2,1,3]),[-1, nItemMax_x, self.num_heads*self.head_size])
        
        # combine multi-head to (nSet*nSet*nItemMax_x, head_size)
        output = self.linearH(weighted_y_Vs)

        if not self.self_attention:
            output = tf.reshape(output,[nSet_x, nItemMax_x, self.head_size])#tf.transpose(tf.reshape(output,[nSet_x, nSet_y, nItemMax_x, self.head_size]),[1,0,2,3])

        else:    
            output = tf.reshape(output,[nSet_x, nItemMax_x, self.head_size])


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
    def __init__(self, isCNN=True, is_set_norm=False, is_cross_norm=True, is_final_linear=True, num_layers=1, num_heads=2, mode='setRepVec_pivot', baseChn=32, rep_vec_num=1, cnn_class_num=2, max_channel_ratio=2, is_neg_down_sample=False):
        super(SMN, self).__init__()
        self.isCNN = isCNN
        self.num_layers = num_layers
        self.mode = mode
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
        # encoder for rep 
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

    # compute score of set-pair using dot product
    def dot_set_score(self, x):
        nSet_x, nSet_y, dim = x.shape
       
        score = tf.stack([[tf.tensordot(x[i,j],x[j,i],1) for i in range(nSet_x)] for j in range(nSet_y)])
        score = tf.expand_dims(score,-1)/tf.cast(dim,float)

        return score

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
            x = self.fc_cnn_proj(x)
            predCNN = []
        
        # reshape (nSet*nItemMax, D) to (nSet, nItemMax, D)
        x = tf.reshape(x,[nSet, nItemMax, -1])

        # # reshape (nSet, nItemMax, D) -> (nSet, nSet, nItemMax, D)
        # x = tf.tile(tf.expand_dims(x,1),[1,nSet,1,1])

        debug['x_cnn'] = x

        # add_embedding
        x_orig = x
        x_rep = tf.tile(self.set_emb, [nSet,1,1])
        x_rep_size = tf.constant(np.full(nSet,self.rep_vec_num).astype(np.float32))

        debug['x_encoder_layer_0'] = x

        x_2enc = x

        #---------------------
        # encoder (self-attention)
        # for query x
        for i in range(self.num_layers):

            z = self.layer_norms_enc1X[i](x,x_size)

            # input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
            z = self.self_attentionsX[i](z,z)
            x += z

            z = self.layer_norms_enc2X[i](x,x_size)
            z = self.fcs_encX[i](z)
            x += z

            debug[f'x_encoder_layer_{i+1}'] = x
        x_enc = x
        #---------------------

        # encoder (self-attention)
        # for rep s
        # for query x
        for i in range(self.num_layers):

            z = self.layer_norms_enc1R[i](x_rep,x_rep_size)

            # input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
            z = self.self_attentionsR[i](z,z)
            x_rep += z

            z = self.layer_norms_enc2R[i](x_rep,x_rep_size)
            z = self.fcs_encR[i](z)
            x_rep += z

        #---------------------

        #---------------------
        # decoder (cross-attention)
        debug[f'x_decoder_layer_0'] = x
        for i in range(self.num_layers):
     
            if self.mode == 'setRepVec_pivot': # Bi-PMA + pivot-cross
                self.cross_attentions[i].pivot_cross = True

            query = self.layer_norms_decq[i](x_rep,x_rep_size) #x_rep = self.layer_norms_enc1[i](x_rep,x_rep_size)
            key = self.layer_norms_deck[i](x,x_size)

            # input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
            query = self.cross_attentions[i](query,key)
            x_rep += query
    
            query = self.layer_norms_dec2[i](x_rep,x_rep_size)
            

            query = self.fcs_dec[i](query)
            x_rep += query

            debug[f'x_decoder_layer_{i+1}'] = x
        x_dec = x
        #---------------------

        #---------------------
        
        # #---------------------

        return predCNN, x_rep, debug

    def dot_set_score(self,x_rep, gallery):
        #pdb.set_trace()
        batch, nSet_rep, dim = x_rep.shape 
        X_flat = tf.reshape(x_rep, [-1,dim])
        Y_flat = tf.reshape(gallery, [-1, dim])
        X_normalized = tf.nn.l2_normalize(X_flat, axis=-1)
        Y_normalized = tf.nn.l2_normalize(Y_flat, axis=-1)

        # 内積を計算
        dot_product = tf.matmul(X_normalized,tf.cast(Y_normalized, dtype=tf.float32), transpose_b=True)

        # 内積の形状を変更して各要素間のCosine類似度を取得
        result = tf.reshape(dot_product, [batch, nSet_rep, gallery.shape[0], nSet_rep]) #一度N_item,N_itemの形に直す
        result = tf.transpose(result,[0,2,1,3]) #(rep_batch,rep_item, gallery_batch, gallery_item )
        beta = 0.2

        for batch_ind in range(len(result)): #一つのクエリに対する代表ベクトル集合とギャラリとの組み合わせループ
            score = result[batch_ind] #あるクエリに対する代表ベクトル集合とギャラリとの類似度マップ

            #バッチ計算分
            score_for_recall = tf.reduce_max(tf.nn.softmax(score,axis=1),axis=1) #tf.reduce_mean(score,axis=1) #RecallでMaxを取る手法だと、maxでない部分の勾配が通らない。だからmean
            #ソフトマックス関数でスコアを計算=> 列ごとに和を取る
            score_for_precision = tf.reduce_max(tf.nn.softmax(score,axis=2),axis=2) #tf.reduce_mean(score,axis=2) #tf.reduce_max(score,axis=2)
            #行ごとに平均を計算
            precision_score = tf.reduce_mean(score_for_precision, axis=1, keepdims=True)
            recall_score = tf.reduce_mean(score_for_recall,axis=1, keepdims=True)
            f1_score = 2*(precision_score*recall_score)/(precision_score+recall_score)
            #f1_score = (precision_score*recall_score*(1+beta**2))/(precision_score+ beta**2 *recall_score)
            if batch_ind==0:
                f1_scores = tf.expand_dims(f1_score,axis=0)
            else:
                f1_scores = tf.concat([f1_scores, tf.expand_dims(f1_score,axis=0)], axis=0)
        y = gallery
        _,nSet_y,_ = y.shape

        return result, f1_scores
    
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
        #クエリとポジティブの位置を設定するために、セットのインデックスを交換する関数
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
        #pdb.set_trace()
        x, y_true = data
        x, x_size = x #xの順番は[a_1,a_2,b_1,b_2, c_1,c_2,...]
        gallery = x #クエリとポジティブを含むすべてのアイテムのデータベース
        positive_item_num = self.get_positive_set_item__num(x_size)
        positive_indices = self.swap_query_positive(gallery)
        #対角成分がpositiveになるようにdatabaseのインデックスを交換
        gallery = tf.gather(gallery, positive_indices)
        # gallery linear projection(dimmension reduction) 
        gallery = self.fc_cnn_proj(gallery)
        with tf.GradientTape() as tape:
            # predict
            predCNN, predSMN, debug = self((x, x_size), training=True)
            
            y_pred = predSMN

            #ポジティブを参照するためのラベル作成
            # y_true = self.set_label(y_true)

            #ポジティブ集合の要素数の参照
            y_true_num = self.get_positive_set_item__num(x_size)

            #conpute similairty with gallery and f1_bert_score
            simiarity, bert_score = self.dot_set_score(predSMN, gallery)

            # # down sampling
            # if self.is_neg_down_sample:
            #     y_true, y_pred = self.neg_down_sampling(y_true, y_pred)

            # loss
            loss = self.compiled_loss(simiarity, bert_score, regularization_losses=self.losses)
     
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
        self.compiled_metrics.update_state(y_true, bert_score)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):
        #pdb.set_trace()
        x, y_true = data
        x , x_size = x
        gallery = x #クエリとポジティブを含むすべてのアイテムのデータベース
        positive_item_num = self.get_positive_set_item__num(x_size)
        positive_indices = self.swap_query_positive(gallery)
        #対角成分がpositiveになるようにdatabaseのインデックスを交換
        gallery = tf.gather(gallery, positive_indices)
        # gallery linear projection(dimmension reduction) 
        gallery = self.fc_cnn_proj(gallery)

        # predict
        predCNN, predSMN, debug = self((x, x_size), training=False)
        y_pred = predSMN

        #ポジティブ集合の要素数の参照
        y_true_num = self.get_positive_set_item__num(x_size)

        #conpute similairty with gallery and f1_bert_score
        simiarity, bert_score = self.dot_set_score(predSMN, gallery)

        # # down sampling
        # if self.is_neg_down_sample:
        #     y_true, y_pred = self.neg_down_sampling(y_true, y_pred)

        # loss
        self.compiled_loss(simiarity, bert_score, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(y_true, bert_score)

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

        simiarity, bert_score = self.dot_set_score(predSMN, gallery)

        return predSMN, simiarity, bert_score
#----------------------------


import argparse
import matplotlib.pylab as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import os
import math
import pdb
import pickle
import seaborn as sns
from PIL import Image


#----------------------------
# model names
def mode_name(mode):
    mode_name = ['maxPooling','poolingMA','CSS','setRepVec_biPMA','setRepVec_pivot']

    return mode_name[mode]
#----------------------------

#----------------------------
# set func names
def calc_set_sim_name(calc_set_sim):
    calc_set_sim_name = ['CS','BERTscore']

    return calc_set_sim_name[calc_set_sim]
#----------------------------

def gallery_type(gallery_type):
    method_name = ['InBatch','OutBatch']

    return method_name[gallery_type]

#----------------------------
def style_method(style_loss):
    method_name = ['not_style','item_style', 'DFA_style']

    return method_name[style_loss]

#----------------------------
# parser for run.py
def parser_run():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('-batch_size', type=int, default=50, help='The number of cordinate sets')
    parser.add_argument('-baseChn', type=int, default=32, help='number of base channel, default=32')
    parser.add_argument('-num_layers', type=int, default=3, help='number of layers (attentions) in encoder and decoder, default=3')
    parser.add_argument('-num_heads', type=int, default=5, help='number of heads in attention, default=5')
    parser.add_argument('-is_set_norm', type=int, default=1, help='switch of set-normalization (1:on, 0:off), default=1')
    parser.add_argument('-trial', type=int, default=1, help='index of trial, default=1')
    parser.add_argument('-train_matching', type=int, default=0, help='Whether training matching model')
    # mlp related
    parser.add_argument('-pretrained_mlp', type=int, default=1, help='Whether pretrain MLP (not use FC_projection)')
    parser.add_argument('-mlp_projection_dim', type=int, default=128, help='MLP hidden last layer projects to the dimension')
    parser.add_argument('-is_Cvec_linear', type=int, default=1, help='Whether learn FC_projection for Cluster seed vec')
    parser.add_argument('-set_loss', type=int, default=0, help='0: set_to_item retrieval, 1: set_to_set retrieval (use CSscore or BERTscore)')
    # dataset 
    parser.add_argument('-tf_data', type=int, default=0, help='0: numpy, 1: tf.data')
    parser.add_argument('-label_ver', type=int, default=1, help='0: use c2_label, 1: use c1_label')
    parser.add_argument('-negative_scheduling', type=int, default=0, help='0: no scheduling negative level is fixed, 1: negative level is increased by epoch')
    parser.add_argument('-category_emb', type=int, default=0, help='0: no category_emb, 1: implement category_emb')

    # loss related
    parser.add_argument('-gallerytype', type=int, default=1, help='0: use in batch negative , 1: use out batch negative')
    parser.add_argument('-style_loss', type=int, default=1, help='0 not use style loss,  1:use style loss , 2: use style loss but with DFA')
    parser.add_argument('-is_l2_loss', type=int, default=0, help='0: not use l2_loss , 1: use l2_loss')

    return parser
#----------------------------

#----------------------------
# parser for comp_results.py
def parser_comp():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='MNIST eventotal matching')
    parser.add_argument('-modes', default='3,4', help='list of score modes, maxPooling:0, poolingMA:1, CSS:2, setRepVec_biPMA:3, setRepVec_pivot:4, default:3,4')
    parser.add_argument('-baseChn', type=int, default=32, help='number of base channel, default=32')
    parser.add_argument('-num_layers', default='3', help='list of numbers of layers (attentions) in encoder and decoder, default=3')
    parser.add_argument('-num_heads', default='5', help='list of numbers of heads in attention, default=5')
    parser.add_argument('-is_set_norm', type=int, default=1, help='switch of set-normalization (1:on, 0:off), default=1')
    parser.add_argument('-is_cross_norm', type=int, default=1, help='switch of cross-normalization (1:on, 0:off), default=1')
    parser.add_argument('-trials', default='1,2,3', help='list of indices of trials, default=1,2,3')
    parser.add_argument('-calc_set_sim', type=int, default=0, help='how to evaluate set similarity, CS:0, BERTscore:1, default=0')
    
    return parser
#----------------------------

#----------------------------
# plot images in specified sets
def plotImg(imgs,set_IDs,msg="",fname="img_in_sets"):
    _, n_item, _, _, _ = imgs.shape
    n_set = len(set_IDs)
    # fig = plt.figure(figsize=(20,5))
    fig = plt.figure()

    for set_ind in range(n_set):                
        for item_ind in range(n_item):
            fig.add_subplot(n_set, n_item, set_ind*n_item+item_ind+1)
            if item_ind == 0:
                plt.title(f'set:{set_IDs[set_ind]}',fontsize=20)
            if item_ind == 1:
                plt.title(f'{msg}',fontsize=20)

            plt.imshow(imgs[set_IDs[set_ind]][item_ind,:,:,0],cmap="gray")
    
    plt.tight_layout()                
    plt.savefig(f'{fname}.png')
#----------------------------

#----------------------------
# plot loss and accuracy
def plotLossACC(path,loss,val_loss,acc,val_acc):
    epochs = np.arange(len(acc))

    fig=plt.figure()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.add_subplot(1,2,1)
    plt.plot(epochs,acc,'bo-',label='training acc')
    plt.plot(epochs,val_acc,'b',label='validation acc')
    plt.title('acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.ylim(0,1)
    
    fig.add_subplot(1,2,2)
    plt.plot(epochs,loss,'bo-',label='training loss')
    plt.plot(epochs,val_loss,'b',label='validation loss')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0,3)
    plt.legend()
    
    result_path = os.path.join(path,"result")
    path = os.path.join(path,"result/loss_acc.png")

    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    plt.savefig(path)
#----------------------------

#----------------------------
# plot histogram
def plotHist(corr_pos, corr_neg, mode, fname=''):
    fig = plt.figure(figsize=(20,5))

    max_data_num = np.max([len(corr_neg[0]),len(corr_pos[0])])
    for layer_ind in range(len(corr_pos)):
        fig.add_subplot(1,len(corr_pos),layer_ind+1)
        plt.hist(corr_neg[layer_ind],label="mismatch",bins=np.arange(-1,1.1,0.1))
        plt.hist(corr_pos[layer_ind],alpha=0.5,label="match",bins=np.arange(-1,1.1,0.1))        
        if layer_ind == 0:
            plt.legend(fontsize=12)
        plt.xlim([-1.2,1.2])
        plt.ylim([0,max_data_num])
        plt.xticks(fontsize=12)

        if layer_ind == 0:
            title = 'input'
        elif layer_ind <= (len(corr_pos)-1)/2:
            title = f'enc{layer_ind}'
        else:
            title = f'dec{layer_ind-(len(corr_pos)-1)/2}'

        plt.title(title)
        
    plt.tight_layout()

    if len(fname):
        plt.savefig(fname)
    else:
        plt.show()
#----------------------------

#----------------------------
# function to compute CMC
def calc_cmcs(pred, true_grp, batch_size, qry_ind=0, glry_start_ind=1, top_n=1):

    # reshape predict and true for each batch
    pred_batchs = np.reshape(pred, [-1, batch_size, batch_size])
    true_grp_batchs = np.reshape(true_grp, [-1, batch_size])

    # extract predicted scores for query and compute true labels 
    pred_scores = pred_batchs[:,qry_ind,glry_start_ind:]

    # label
    true_labs = (true_grp_batchs == true_grp_batchs[:,[qry_ind]])[:,glry_start_ind:].astype(int)

    # shuffle pred and true
    np.random.seed(0)
    random_inds = random_inds = np.vstack([np.random.permutation(len(true_labs[0])) for i in range(len(true_labs))]) 
    pred_scores = np.vstack([pred_scores[i][random_inds[i]] for i in range(len(random_inds))])
    true_labs = np.vstack([true_labs[i][random_inds[i]] for i in range(len(random_inds))])

    # sort predicted scores and compute TP map (data x batch_size)
    pred_sort_inds = np.argsort(pred_scores,axis=1)[:,::-1]
    TP_map = np.take_along_axis(true_labs,pred_sort_inds,axis=1)

    cmcs = np.sum(np.cumsum(TP_map,axis=1),axis=0)/len(true_labs)

    return cmcs
#----------------------------

def Set_hinge_loss(scores:tf.Tensor, y_true:tf.Tensor)->tf.Tensor:

    """Loss function for Set Retrieval task
    In for loop, getting set similarity score (between pred and gallery y)
    , getting positive score and the others by y_true (set labels, which identify positive position), and calculating hingeloss """
    
    Set_hinge_losssum = []
    slack_variable = 0.4
    # scores: (nSet_x, nSet_y), y_true: (nSet_x, nSet_y)
    for batch_ind in range(len(y_true)): 
        
        # score , bert_score or CS score between \hat y [batch_ind] and y : (nSet_y,) 
        score = scores[batch_ind]
        # positive_score : (1, )
        # negative_score : (nSet_y - 1, )
        positive_score = tf.boolean_mask(score, tf.equal(y_true[batch_ind], 1))
        negative_score = tf.boolean_mask(score, tf.equal(y_true[batch_ind], 0))

        # hingeloss : (nSet_y - 1, )
        hingeloss = tf.maximum(negative_score - positive_score + slack_variable , 0.0)
        Set_hinge_losssum.append(tf.reduce_sum(hingeloss))
        

    Loss = sum(Set_hinge_losssum)/len(Set_hinge_losssum)
    
    return Loss
#----------------------------

def create_true_index(size):
    # 初期化 - 単位行列を作成
    matrix = tf.eye(size, dtype=tf.float32)

    # 隣接するインデックスの入れ替え
    for i in range(0, size, 2):
        if i + 1 < size:
            matrix = tf.tensor_scatter_nd_update(matrix, [[i, i+1], [i+1, i]], [1.0, 1.0])
            matrix = tf.tensor_scatter_nd_update(matrix, [[i, i], [i+1, i+1]], [0.0, 0.0])
    
    return matrix
def swap_query_positive(array):
    #クエリとポジティブの位置を設定するために、セットのインデックスを交換する関数
    indices = tf.range(0, tf.shape(array)[0])
    swapped_indices = tf.reshape(tf.stack([indices[1::2], indices[::2]], axis=-1), [-1])

    return swapped_indices
# In Batch negative loss (zozo + alpha) 
def InBatchCLIPLoss(labels:tf.Tensor,  cos_sim: tf.Tensor)->tf.Tensor:
    
    y_true = create_true_index(labels.shape[0])
    set_size = tf.reduce_sum(tf.cast(labels != tf.reduce_max(labels), tf.float32), axis=1)
    pred_set_size = tf.gather(set_size, tf.where(tf.equal(y_true,1))[:,1])

    negative_sampling = "mean"
    threshold_alpha = 0.10
    # Switching CLPPNeg option manually...
    CLPPNeg = True
    # -------------------------------------
    
    batch_loss = []
    # bce = tf.losses.BinaryCrossentropy(from_logits=False)
    for batch_ind in range(labels.shape[0]):
        item_loss = []
        for item_ind in range(labels.shape[1]):
            target_label = labels[tf.where(tf.equal(y_true,1))[:,1][batch_ind]][item_ind] # 予測カテゴリの取得
            if target_label == tf.reduce_max(labels):# 0パディングの時
                item_loss.append(0)
            else:
                positive_score = cos_sim[batch_ind, :, item_ind, item_ind][tf.where(y_true==1)[:,1][batch_ind]]
                indices = tf.where(labels == target_label) # 同じカテゴリのアイテムインデックスを検索
                if len(indices[:, 0]) == 1: # 同じカテゴリのnegativeがない場合
                    padding_mask = tf.cast(labels!=41,tf.float32)
                    # all scores
                    pred_scores = tf.reshape(cos_sim[batch_ind, :, item_ind, :],-1)
                    
                    label = tf.zeros((labels.shape[0], labels.shape[1]), dtype=tf.float32)
                    label = tf.tensor_scatter_nd_update(label, [[indices[:,0].numpy()[0], item_ind]], [1])
                    label = tf.reshape(label, -1)
                    
                    # losses manually
                    # Extract positive scores using Label: (Batch, N)
                    positive_scores = tf.reduce_sum(pred_scores * label)
                    
                    # Mask out positive positions for negatives: (Batch, N, Batch)
                    negative_mask = 1.0 - label
                    negative_scores = pred_scores * negative_mask

                    negative_scores = negative_scores * tf.reshape(padding_mask,-1)

                    if negative_sampling == "max":
                        negative_scores = tf.reduce_max(negative_scores)
                    elif negative_sampling == "top5":
                        if len(negative_scores) > 5:
                            negative_scores = tf.math.top_k(negative_scores, k=5).values
                    
                    # Calculate exponential scores for negative normalization: (Batch, N, Batch)
                    negative_exp = tf.exp(negative_scores)
                    
                    # Calculate exponential scores for negative normalization: (Batch, N, Batch)
                    # negative_exp = tf.exp(negative_scores)
                    positive_exp = tf.exp(positive_scores)
                    
                    # Denominator includes positive and all negative scores: (Batch, N)
                    denom = tf.reduce_mean(negative_exp) + positive_exp
                    
                    # Contrastive loss for each item: (Batch, N)
                    entropy_loss = -tf.math.log(positive_exp / denom)
                    item_loss.append(entropy_loss)
                    # item_loss.append(0)
                else:
                    true_indices = tf.argmax(y_true, axis=1)
                    pred_scores = tf.stack([
                        cos_sim[batch_ind, indices[i, 0], item_ind, indices[i, 1]]
                        for i in range(indices.shape[0])
                    ])
                    label = tf.cast(tf.equal(indices[:,0], true_indices[batch_ind]), tf.float32)
                    
                    # contrastive loss
                    if sum(label) >= 2: #Zのラベルが重複している場合positive_scoreと一致しているものが真値
                        label = tf.cast(tf.equal(pred_scores, positive_score), tf.float32)
                    # losses manually
                    # Extract positive scores using Label: (Batch, N)
                    positive_scores = tf.reduce_sum(pred_scores * label)
                
                    # Mask out positive positions for negatives: (Batch, N, Batch)
                    negative_mask = 1.0 - label
                    negative_scores = pred_scores * negative_mask

                    if negative_sampling == "max":
                        negative_scores = tf.reduce_max(negative_scores)
                    elif negative_sampling == "top5":
                        if len(negative_scores) > 5:
                            negative_scores = tf.math.top_k(negative_scores, k=5).values
                    
                    # Calculate exponential scores for negative normalization: (Batch, N, Batch)
                    negative_exp = tf.exp(negative_scores)
                    positive_exp = tf.exp(positive_scores)
                    
                    if CLPPNeg:
                        score_diff = positive_exp - negative_exp
                        mask  = tf.cast(score_diff < positive_exp * threshold_alpha, tf.float32)
                        masked_negative_exp = negative_exp * mask
                        negative_exp_sum = tf.reduce_sum(masked_negative_exp)
                        denom = negative_exp_sum + positive_exp
                    else:
                        # Denominator includes positive and all negative scores: (Batch, N)
                        negative_exp_sum = tf.reduce_mean(negative_exp)
                        denom = negative_exp_sum + positive_exp
                    
                    # Contrastive loss for each item: (Batch, N)
                    entropy_loss = -tf.math.log(positive_exp / denom)
                    # entropy_loss = bce(label, pred_scores)
                    item_loss.append(entropy_loss)
                    
                    if tf.math.is_nan(positive_score):
                        pdb.set_trace()
                    
        batch_loss.append(sum(item_loss) / pred_set_size[batch_ind])

    Loss = tf.stack(batch_loss)
    
    return Loss

def OutBatchCLIPLoss(labels:tf.Tensor,  cos_sim: tf.Tensor)->tf.Tensor:
    
    y_true = create_true_index(labels.shape[0])
    set_size = tf.reduce_sum(tf.cast(labels != tf.reduce_max(labels), tf.float32), axis=1)
    pred_set_size = tf.gather(set_size, tf.where(tf.equal(y_true,1))[:,1])
    batch_size, item_size, score_size = cos_sim.shape

    threshold_alpha = 0.10
    # Switching CLPPNeg option manually...
    CLPPNeg = True

    # One-hotラベルを計算するためのマスク作成
    range_items = tf.range(item_size, dtype=tf.float32)  # (nItemMax,)
    item_mask = tf.expand_dims(pred_set_size, axis=1) > range_items  # Shape: (Batch, nItemMax)
    # cos_sim : (Batch, nItemMax, nPositive + nNegativeMax, D)
    # Positiveスコア: 各アイテム方向の最初の要素
    positive_scores = cos_sim[:, :, 0]  # Shape: (Batch, nItemMax)
    
    # Negativeスコア: 残りの部分
    negative_scores = cos_sim[:, :, 1:]  # Shape: (Batch, nItemMax, nNegativeMax)
    
    # Positiveスコアの指数
    positive_exp = tf.exp(positive_scores)  # Shape: (Batch, nItemMax)
    
    # Negativeスコアの指数
    negative_exp = tf.exp(negative_scores) # Shape: (Batch, nItemMax, nNegativeMax)

    # expPositiveとexpNegativeスコアの差を計算
    score_diff = tf.expand_dims(positive_exp, axis=2) - negative_exp
    # スコア差がξ未満の箇所を残すマスク
    mask = tf.cast(score_diff < (tf.expand_dims(positive_exp,axis=-1) * threshold_alpha), tf.float32) # Shape: (Batch, item_size, nNegativeMax)
    masked_negative_exp = negative_exp * mask  # Masked Negative scores
    # Negativeスコアの指数の和
    if CLPPNeg:
        negative_exp_sum = tf.reduce_sum(masked_negative_exp, axis=2)  # Shape: (Batch, nItemMax)
    else:
        negative_exp_sum = tf.reduce_sum(negative_exp, axis=2)  # Shape: (Batch, nItemMax)

    # 分母: positive + negatives
    denom = positive_exp + negative_exp_sum  # Shape: (Batch, nItemMax)

    # 損失計算: 各アイテム方向
    log_prob_positive = tf.math.log(positive_exp / denom)  # Shape: (Batch, nItemMax)
    item_loss = -log_prob_positive  # Shape: (Batch, nItemMax)

    # 必要なアイテムだけ計算 (予測サイズ以上はマスク処理)
    masked_item_loss = item_loss * tf.cast(item_mask, tf.float32)  # Shape: (Batch, nItemMax)

    # アイテム方向の平均
    batch_loss = tf.reduce_sum(masked_item_loss, axis=1) / tf.maximum(pred_set_size, 1.0)  # Shape: (Batch,)
    
    return batch_loss
def Category_accuracy(pred_labels, true_labels):
    # まず、41を無視したマスクを作成
    mask = tf.not_equal(true_labels, 41)  # 41でない部分がTrue
    
    # マスクを使用して、予測ラベルと正解ラベルを比較
    correct_predictions = tf.equal(pred_labels, true_labels)
    
    # マスクされた部分のみの一致率を計算
    correct_predictions_masked = tf.boolean_mask(correct_predictions, mask)
    mask_per_row = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1)  # 行ごとの有効なラベル数
    correct_per_row = tf.reduce_sum(tf.cast(correct_predictions, tf.float32) * tf.cast(mask, tf.float32), axis=1)
    
    # 行ごとの正解率を計算（無視された要素は分母に入らない）
    accuracy_per_row = correct_per_row / mask_per_row
    
    # 行方向の平均を取り、全体の正解率を求める
    overall_accuracy = tf.reduce_mean(accuracy_per_row)
    
    return overall_accuracy

def Set_accuracy(score, y_true):
    """Custom Metrics Function to evaluate set similarity between pred item set \hat y and gallery y"""
    """1 : positive_score is in top10 % of set similarity pairs , 0 : otherwise"""
    # threshold K 
    threk = int(len(score)*0.01)
    
    accuracy = np.zeros((len(score), 1))

    for batch_ind in range(len(score)):
        f1_score = score[batch_ind]
        _, topscore_index = tf.nn.top_k(f1_score, k=threk)
        if tf.where(tf.equal(y_true[batch_ind], 1))[0].numpy() in topscore_index: # (tf.where(tf.equal(y_true[batch_ind], 1))[0].numpy()) finds positive index.
            accuracy[batch_ind] += 1

    return accuracy

def matching_accuracy(y_true, y_pred):
    return tf.reduce_mean(y_pred)

def Retrieval_acc(scores, needed_items):
    """
    Calculate accuracy based on whether the positive score is within the top 3 scores.

    Args:
        scores: Tensor of shape (Batch, Items, Scores), representing score arrays.
        needed_items: Tensor of shape (Batch,), indicating the number of valid items for each batch.

    Returns:
        accuracy: Scalar tensor representing the mean accuracy across the batch.
    """
    
    '''# Step 1: Extract positive scores (first score in each row of items)
    positive_scores = scores[:, :, 0]  # Shape: (Batch, Items)

    # Step 2: Sort scores along the last axis in descending order
    sorted_scores = tf.sort(scores, axis=2, direction="DESCENDING")  # Shape: (Batch, Items, Scores)

    # Step 3: Find the rank of positive scores among the top 3 scores
    top_3_scores = sorted_scores[:, :, :3]  # Top 3 scores: Shape (Batch, Items, 3)
    positive_in_top_3 = tf.reduce_any(
        tf.expand_dims(positive_scores, axis=2) == top_3_scores, axis=2
    )  # Shape: (Batch, Items), True where positive score is in top 3

    # Step 4: Apply mask using needed_items to exclude invalid items
    mask = tf.sequence_mask(needed_items, maxlen=tf.shape(scores)[1])  # Shape: (Batch, Items)
    valid_accuracies = tf.cast(positive_in_top_3, tf.float32) * tf.cast(mask, tf.float32)  # Mask invalid items

    # Step 5: Calculate mean accuracy per batch (ignoring invalid items)
    batch_accuracies = tf.reduce_sum(valid_accuracies, axis=1) / tf.maximum(
        tf.cast(needed_items, tf.float32), 1.0
    )  # Prevent division by zero

    # Step 6: Calculate the overall accuracy across all batches
    overall_accuracy = tf.reduce_mean(batch_accuracies)  # Scalar
    '''

    # Step 1: Extract positive scores (first score in each row of items)
    positive_scores = scores[:, :, 0]  # Shape: (Batch, Items)

    # Step 2: Sort scores along the last axis in descending order
    sorted_scores = tf.sort(scores, axis=2, direction="DESCENDING")  # Shape: (Batch, Items, Scores)

    # Step 3: Determine the ranks of positive scores
    # Compare positive scores to the sorted scores to determine their ranks
    is_positive_rank = tf.cast(
        tf.expand_dims(positive_scores, axis=2) == sorted_scores, tf.float32
    )  # Shape: (Batch, Items, Scores), 1 where scores match
    ranks = tf.reduce_sum(is_positive_rank * tf.range(1, tf.shape(scores)[2] + 1, dtype=tf.float32), axis=2)  
    # Shape: (Batch, Items), rank for each positive score (1-based)

    # Step 4: Apply mask using needed_items to exclude invalid items
    mask = tf.sequence_mask(needed_items, maxlen=tf.shape(scores)[1])  # Shape: (Batch, Items)
    valid_ranks = ranks * tf.cast(mask, tf.float32)  # Mask invalid items

    # Step 5: Calculate mean rank per batch (ignoring invalid items)
    batch_mean_ranks = tf.reduce_sum(valid_ranks, axis=1) / tf.maximum(
        tf.cast(needed_items, tf.float32), 1.0
    )  # Prevent division by zero 

    # Step 6: Calculate the overall average rank across all batches
    overall_average_rank = tf.reduce_mean(batch_mean_ranks)  # Scalar

    return overall_average_rank
#----------------------------
# convert class labels to cross-set label（if the class-labels are same, 1, otherwise 0)
def cross_set_label(y):
    # rows of table
    y_rows = tf.tile(tf.expand_dims(y,-1),[1,tf.shape(y)[0]])
    # cols of table       
    y_cols = tf.tile(tf.transpose(tf.expand_dims(y,-1)),[tf.shape(y)[0],1])

    # if the class-labels are same, 1, otherwise 0
    labels = tf.cast(y_rows == y_cols, float)            
    return labels

def gram_matrix(input_tensor):
    # result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

    result = tf.einsum('bnd,bne->bnde', input_tensor, input_tensor) # pixel loss
    # result = tf.einsum('bnd,bnd->bn', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)
def compute_test_rank(gallery, pred, set_label, category):
    
    y_true = cross_set_label(set_label)
    y_true = tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))
    # y_true = y_true[:100]
    
    nItemMax_y = gallery.shape[1]
    nItemPred = pred.shape[1]
    x_norm = gallery / np.linalg.norm(gallery, axis=-1, keepdims=True)
    y_norm = pred / np.linalg.norm(pred, axis=-1, keepdims=True)

    # x_norm = x_norm[:100]
    # y_norm = y_norm[:100]
    
    nItem = np.count_nonzero(~np.all(gallery == 0, axis=-1), axis=1)

    # nItem = nItem[:100]

    # コサイン類似度の計算
    x_expand = np.expand_dims(x_norm, 0) # (nSet_x, 1, nItemMax, head_size)
    y_expand = np.expand_dims(y_norm, 1) # (1, nSet_y, nItemMax, head_size)
    cos_sim = np.einsum('aijk,ibmk->abjm', y_expand, x_expand, optimize='optimal') # (nSet_y, nSet_x, nItemMax_x, nItemMax_y)

    def find_ranks(array, targets):
        valid_indices = np.where(~np.isnan(array))[0]
        valid_array = array.numpy()[valid_indices]
        sorted_indices = np.argsort(valid_array)[::-1]
        sorted_array = valid_array[sorted_indices]
        
        if len(array.shape) ==1:
            for target in targets:
                if math.isnan(target):
                    ranks = -1
                else:
                    rank = np.where(sorted_array == target)
                    if len(rank) > 0:
                        ranks = rank[0][0] + 1  # インデックスを1ベースに変換し、最も低いランクを追加
                    else:
                        ranks = -1  # 要素が見つからない場合は-1を追加
        else:
            ranks = []
            for target in targets:
                if math.isnan(target):
                    ranks.append(-1)
                else:
                    rank = np.where(sorted_array == target)
                    if len(rank) > 0:
                        ranks.append(rank[0][0] + 1)  # インデックスを1ベースに変換し、最も低いランクを追加
                    else:
                        ranks.append(-1)  # 要素が見つからない場合は-1を追加

        return ranks
    ranks = []
    for batch_ind in range(len(cos_sim)):
        rank = []
        for item_ind in range(nItemPred):
            if category[tf.where(tf.equal(y_true[batch_ind], 1))[0][0].numpy()][item_ind] != 11:
                indices = tf.where(category == category[tf.where(tf.equal(y_true[batch_ind], 1))[0][0].numpy()][item_ind])
                if len(indices[:,0]) == 1:
                    rank.append(1)
                else:  
                    true_score = [cos_sim[batch_ind][:,item_ind,:][tf.where(tf.equal(y_true[batch_ind], 1))[0][0].numpy(), item_ind]]
                    target_cos_sim = tf.stack([cos_sim[batch_ind][:,item_ind,:][indices.numpy()[i, 0], indices.numpy()[i, 1]] for i in range(indices.shape[0])])

                    reshaped_array = target_cos_sim
                    
                    # target_score = tf.gather_nd(cos_sim[batch_ind][:,item_ind,:], tf.where(tf.equal(y_true[batch_ind], 1))).numpy()[0].tolist()
                    rank.append(find_ranks(reshaped_array, true_score))
            else:
                rank.append(-1)

        ranks.append(rank)
    
    return ranks
#----------------------------   

def ranking_analysis(Ranking_pkl_path, data_path, Dataset='DeepFurniture'):
    # Roading Result 
    if Dataset == 'Shift15M':
        with open(Ranking_pkl_path, 'rb') as fp:
            rank_result = pickle.load(fp)

        with open(data_path, 'rb') as fp:
            test_pred_vec = pickle.load(fp)
            gallery = pickle.load(fp)
            y_test = pickle.load(fp)
            replicated_set_label = pickle.load(fp)
            query_id = pickle.load(fp)
            c_label_test = pickle.load(fp)
            item_label_test = pickle.load(fp)
        MAXCATEGORY = 41
    elif Dataset == 'DeepFurniture':
        with open(Ranking_pkl_path, 'rb') as fp:
            rank_result = pickle.load(fp)
        with open(data_path, 'rb') as fp:
            test_pred_vec = pickle.load(fp)
            gallery = pickle.load(fp)
            y_test = pickle.load(fp)
            # replicated_set_label = pickle.load(fp)
            # query_id = pickle.load(fp)
            c_label_test = pickle.load(fp)
            item_label_test = pickle.load(fp)
        MAXCATEGORY = 11

    y_true = cross_set_label(y_test)
    y_true = tf.linalg.set_diag(y_true, tf.zeros(y_true.shape[0], dtype=tf.float32))
    c_label_test = tf.gather(c_label_test, tf.where(y_true==1)[:,1])

    rank_array = np.array(rank_result)
    unique_labels = np.unique(c_label_test)

    # カテゴリごとのランク, スコアを格納する辞書
    score_dict = {i: [] for i in range(MAXCATEGORY)}
    acc_dict = {i: [] for i in range(MAXCATEGORY)}

    if Dataset == 'Shift15M':
        label_to_index = {'0': 41, '10001': 0, '10002': 1, '10003': 2, '10004': 3, '10005': 4, '11001': 5, '11002': 6, '11003': 7, '11004': 8, '11005': 9, '11006': 10, '11007': 11, '11008': 12, '12001': 13, '12002': 14, '12003': 15, '12004': 16, '12005': 17, '13001': 18, '13002': 19, '13003': 20, '13004': 21, '13005': 22, '14001': 23, '14002': 24, '14003': 25, '14004': 26, '14005': 27, '14006': 28, '14007': 29, '15001': 30, '15002': 31, '15003': 32, '15004': 33, '15005': 34, '15006': 35, '15007': 36, '16001': 37, '16002': 38, '16003': 39, '16004': 40}

        vectorized_mapping = np.vectorize(lambda x: label_to_index[str(x)])

        c_label_test = vectorized_mapping(c_label_test.numpy())
    elif Dataset == 'DeepFurniture':
        c_label_test = c_label_test.numpy()
        
    K_Threshold = [0.01, 0.05, 0.10, 0.20]

    # カテゴリごとにランクを辞書に格納
    for labels, score_row in zip(c_label_test, rank_array):
        for label, score in zip(labels, score_row):
            if not label >= MAXCATEGORY:
                if label != MAXCATEGORY:  # パディングクラスを無視
                    score_dict[label].append(score)

    for k in K_Threshold:
        topK_score = (np.unique(c_label_test, return_counts=True)[1]*k).astype(np.int64)
        for label in range(MAXCATEGORY):
            binary_data = [1 if x < topK_score[label] else 0 for x in score_dict[label]]
            acc_dict[label].append(binary_data)

        sum = []
        for i in range(MAXCATEGORY):
            sum.append(np.array(acc_dict[i]).mean())
        print(f"Top{int(k*100)}percentile Accuracy (std) : ", np.array(sum).mean(), np.array(sum).std())
        acc_dict = {i: [] for i in range(MAXCATEGORY)}

    # # カテゴリごとにランクの平均を計算
    # label_means = {}
    # for label in unique_labels:
    #     mask = (c_label_test == label)
    #     if label==0:
    #         mean_score = 0
    #     else:
    #         mean_score = rank_array[mask].mean()
    #     label_means[label] = mean_score
    #     bin_count = int(np.sqrt(len(rank_array[mask])))


def visualize_set_top3(
    data_path,
    image_dir, 
    save_dir
):
    """
    predをクエリ、galleryをデータベースとみなし、カテゴリは考慮せず、
    各セット (i=0～nSet-1) を1枚の画像 (行=4,列=nItem=8) で描画する。
    
    - 行0: クエリ画像 (pred[i,j])。もし test_item_id[i,j]==0 なら "Mask" としてアイテム無し。
    - 行1～3: Top1～Top3候補。自分自身含む全galleryアイテム(nSet×nItem)との類似度をランキングし、上位3件を表示。
    - 類似度はcosine類似度(L2正規化後の内積)。
    - ファイル名 "set_1.png","set_2.png",…の連番。
    - ans_c_label[i]はセット単位のラベル。例としてFigure全体のタイトルに表示。
    - カテゴリには依存しない(=カテゴリによる除外しない)ので、Maskでもtest_item_id!=0ならTop3に出てくる可能性あり。
    - 灰色の縦線は描画しない(ユーザ要望)。
    """
    with open(data_path, 'rb') as fp:
        pred = pickle.load(fp) # (nSet, nItem, d) : test_pred_vec
        gallery = pickle.load(fp) # (nSet, nItem, d) : gallery
        y_test = pickle.load(fp)
        c_label_test = pickle.load(fp)
        ans_c_label = pickle.load(fp) # (nSet,)          : 1セットごとのラベル(例: ans_c_label_test)
        test_item_id = pickle.load(fp) # (nSet, nItem)    : ID(0なら実在しない => "Mask")

    os.makedirs(save_dir, exist_ok=True)

    nSet, nItem, d = gallery.shape
    eps = 1e-10

    # 1) L2正規化
    pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + eps)     # shape=(nSet, nItem, d)
    gal_norm  = gallery / (np.linalg.norm(gallery, axis=-1, keepdims=True) + eps)

    # 2) cos_simの一括計算: shape=(nSet_pred, nSet_gallery, nItem_pred, nItem_gallery)
    #    cos_sim[i, x, j, y] = < pred_norm[i,j], gal_norm[x,y] >
    x_expand = np.expand_dims(gal_norm, 0)   # (1, nSet_x, nItem, d)
    y_expand = np.expand_dims(pred_norm, 1)  # (nSet_y, 1, nItem, d)
    cos_sim = np.einsum('aijk,ibmk->abjm', y_expand, x_expand, optimize='optimal')
    # shape = (nSet, nSet, nItem, nItem)

    set_counter = 1

    for i in range(nSet):
        # Figure (行=4, 列=nItem)
        fig, axes = plt.subplots(11, nItem, figsize=(3*nItem, 12))
        # Figureタイトルに "Set {i} => ans_c_label[i]"
        if ans_c_label is not None and i < len(ans_c_label):
            fig.suptitle(f"Set {i}: Category={ans_c_label[i]}", fontsize=14)

        for col in range(nItem):
            q_id = test_item_id[i, col]
            if q_id == 0:
                # Mask or 非存在アイテム => 1行目に "Mask"
                axes[0, col].text(0.5, 0.5, "Mask", ha="center", va="center", fontsize=12)
                axes[0, col].axis("off")
                # 下3行も "Mask"
                for r in range(1,11):
                    axes[r, col].text(0.5, 0.5, "Mask", ha="center", va="center")
                    axes[r, col].axis("off")
                continue
            else:
                # クエリ画像 (pred[i,col]) の表示
                q_img_path = os.path.join(image_dir, f"{q_id}.jpg")
                try:
                    q_img = Image.open(q_img_path)
                except:
                    q_img = None
                # axes[0, col].set_title(f"Positive")  
                if q_img is not None:
                    axes[0, col].imshow(q_img)
                else:
                    axes[0, col].text(0.5, 0.5, "Image not found", ha="center", va="center")
                axes[0, col].axis("off")

                # # 3) Top3計算: cos_sim[i, :, col, :] => shape=(nSet,nItem)
                # vals_2d = cos_sim[i, :, col, :]     # shape=(nSet, nItem)
                # vals = vals_2d.ravel()              # shape=(nSet*nItem,)

                # # Top3を取得 (自分自身含む)
                # top_idx = np.argsort(vals)[-3:][::-1]

                # 同一カテゴリの候補（自身を除く）
                same_cat_mask = (ans_c_label == ans_c_label[i,col])
                
                same_cat_mask[i] = False  # 自身除外
                candidate_indices = np.where(same_cat_mask)[0]
                
                if candidate_indices.size == 0:
                    print(f"Positive index {q_idx}: 同一カテゴリの候補が存在しない")
                    continue
                
                vals_2d = cos_sim[i][candidate_indices][:,col,:] # (N(L), nItem)
                vals_2d = vals_2d[same_cat_mask[candidate_indices]]
                vals = vals_2d.ravel()              # shape=(N(L),)
                
                # 上位3候補のインデックス（候補集合内での相対インデックス）
                top_idx = np.argsort(vals)[-10:][::-1]
                
                # top3_global_idx = candidate_indices[top3_rel_idx]
                for r, tidx in enumerate(top_idx):
                    sim_val = vals[tidx]
                    set_x  = tidx // nItem   # flattenでのセット番号
                    item_x = tidx %  nItem   # flattenでのアイテム番号
                    # cand_id = test_item_id[set_x, item_x]
                    cand_id =  test_item_id[candidate_indices][same_cat_mask[candidate_indices]]
                    cand_id = cand_id.ravel()
                    cand_id = cand_id[tidx]

                    cand_path = os.path.join(image_dir, f"{cand_id}.jpg")
                    try:
                        cand_img = Image.open(cand_path)
                    except:
                        cand_img = None
                    # axes[r+1, col].set_title(f"Top{r+1}")
                    if cand_img is not None:
                        axes[r+1, col].imshow(cand_img)
                    else:
                        axes[r+1, col].text(0.5, 0.5, "Image not found", ha="center", va="center")
                    axes[r+1, col].axis("off")

        plt.tight_layout()

        # 灰色の線はユーザ要望で描画しない

        save_path = os.path.join(save_dir, f"set_{set_counter}.png")
        plt.savefig(save_path)
        plt.close(fig)

        set_counter += 1

    print("Visualization done.")

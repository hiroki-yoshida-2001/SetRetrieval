import argparse
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import os
import math
import pdb


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
    # mlp related
    parser.add_argument('-pretrained_mlp', type=int, default=1, help='Whether pretrain MLP (not use FC_projection)')
    parser.add_argument('-mlp_projection_dim', type=int, default=128, help='MLP hidden last layer projects to the dimension')
    parser.add_argument('-is_Cvec_linear', type=int, default=1, help='Whether learn FC_projection for Cluster seed vec')
    parser.add_argument('-use_all_pred', type=int, default=0, help='0: set_to_item retrieval, 1: set_to_set retrieval')
    # dataset 
    parser.add_argument('-tf_data', type=int, default=0, help='0: numpy, 1: tf.data')
    parser.add_argument('-label_ver', type=int, default=1, help='0: use c2_label, 1: use c1_label')
    
    parser.add_argument('-category_emb', type=int, default=0, help='0: no category_emb, 1: implement category_emb')

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

def Set_item_Cross_entropy(labels:tf.Tensor,  cos_sim: tf.Tensor)->tf.Tensor:

    # 正解集合の特定に問題あり: y_true (set_label) を使って探すべき,..., 2変数しか入力できない
    # 隣のインデックスがセットになっていると仮定
    y_true = create_true_index(labels.shape[0])
    set_size = tf.reduce_sum(tf.cast(labels != 41, tf.float32), axis=1)
    pred_set_size = tf.gather(set_size, tf.where(tf.equal(y_true,1))[:,1])
    # -------------------------------------
    cce = tf.losses.CategoricalCrossentropy()
    
    batch_loss = []
    for batch_ind in range(labels.shape[0]):
        item_loss = []
        for item_ind in range(labels.shape[1]):
            target_cos_sim = cos_sim[batch_ind,:, item_ind, :]

            target_label = labels[tf.where(tf.equal(y_true,1))[:,1][batch_ind]][item_ind]

            if target_label == 41:
                item_loss.append(0)
            else:
                binary_labels = tf.where(labels == target_label, 1.0, 0.0)

                entropy_loss = cce(tf.reshape(binary_labels,-1), tf.reshape(target_cos_sim, -1))

                item_loss.append(entropy_loss)
        batch_loss.append(sum(item_loss) / pred_set_size[batch_ind])
    
    return sum(batch_loss)/len(batch_loss)

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
    pdb.set_trace()
    
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
            if category[tf.where(tf.equal(y_true[batch_ind], 1))[0][0].numpy()][item_ind] != 0:
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
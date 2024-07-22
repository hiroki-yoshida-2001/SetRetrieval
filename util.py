import argparse
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import os
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
    parser.add_argument('-mode', type=int, default=0, help='mode of computing set-matching score, maxPooling:0, poolingMA:1, CSS:2, setRepVec_biPMA:3, setRepVec_pivot:4, default:0')
    parser.add_argument('-baseChn', type=int, default=32, help='number of base channel, default=32')
    parser.add_argument('-num_layers', type=int, default=3, help='number of layers (attentions) in encoder and decoder, default=3')
    parser.add_argument('-num_heads', type=int, default=5, help='number of heads in attention, default=5')
    parser.add_argument('-is_set_norm', type=int, default=1, help='switch of set-normalization (1:on, 0:off), default=1')
    parser.add_argument('-is_cross_norm', type=int, default=1, help='switch of cross-normalization (1:on, 0:off), default=1')
    parser.add_argument('-trial', type=int, default=1, help='index of trial, default=1')
    parser.add_argument('-calc_set_sim', type=int, default=0, help='how to evaluate set similarity, CS:0, BERTscore:1, default=0')
    parser.add_argument('-use_Cvec', type=int, default=1, help='Whether use Cvec')
    parser.add_argument('-is_Cvec_linear', type=int, default=1, help='Whether divide FC_projection for Cluster seed vec and query')
    parser.add_argument('-pretrained_mlp', type=int, default=1, help='Whether pretrain MLP (not use FC_projection)')
    parser.add_argument('-mlp_projection_dim', type=int, default=128, help='MLP will be learned to achieve designated dimention')

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
    slack_variable = 0.2
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
        Set_hinge_losssum.append(tf.reduce_mean(hingeloss))
        

    Loss = sum(Set_hinge_losssum)/len(Set_hinge_losssum)
    
    return Loss
#----------------------------

def Set_accuracy(score:tf.Tensor, y_true:tf.Tensor):
    
    """Custom Metrics Function to evaluate set similarity between pred item set \hat y and gallery y"""
    """1 : positive_score is in top10 % of set similarity pairs , 0 : otherwise"""
    # threshold K 
    threk = int(len(score)*0.1)
    
    accuracy = np.zeros((len(score), 1))

    for batch_ind in range(len(score)):
        f1_score = score[batch_ind]
        _, topscore_index = tf.nn.top_k(f1_score, k=threk)
        if tf.where(tf.equal(y_true[batch_ind], 1))[0].numpy() in topscore_index: # (tf.where(tf.equal(y_true[batch_ind], 1))[0].numpy()) finds positive index.
            accuracy[batch_ind] += 1

    return accuracy

#----------------------------

# compute item wise similarity between pred Item and gallery and select item(in predict step)
def item_selection(x, ID):
    x, y = x # x, y : (nSet_x(y), nItemMax, dim)
    category2, item_label, set_label = ID
    nSet_x = tf.shape(x)[0]
    nSet_y = tf.shape(y)[0]
    nItemMax_y = tf.shape(y)[1]
    
    x_norm = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=-1, keepdims=True)
    
    # コサイン類似度の計算
    cos_sim = np.stack(
        [
            [
                np.dot(y_norm[j], x_norm[i].T)
                for i in range(len(x))
            ]
            for j in range(len(y))
        ]
    )
    
    # category2 label wise methods
    expected_category2 = tf.constant([10001, 10002, 10003, 10004, 10005, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008
                        , 12001, 12002, 12003, 12004, 12005, 13001, 13002, 13003, 13004, 13005
                        , 14001, 14002, 14003, 14004, 14005, 14006, 14007, 15001, 15002, 15003, 15004, 15005, 15006, 15007
                        , 16001, 16002, 16003, 16004])
    pred_id_data_batch = []
    pred_id_data = []
    for batch_ind in range(len(cos_sim)):
        for category2_ind in expected_category2:
            indices = tf.where(category2 == category2_ind.numpy())
            target_cos_sim = tf.stack([cos_sim[batch_ind][indices[i, 0], :, indices[i, 1]] for i in range(indices.shape[0])])
            if len(target_cos_sim) == 0:
                target_cos_sim = 0
                pred_set_label = 0
                pred_item_label = 0
                pred_set_label = tf.cast(pred_set_label, tf.int64)
                pred_item_label = tf.cast(pred_item_label, tf.int64)
            else:
                # 各ラベル毎の最近傍のIDさえ返せればいいです
                flattened = tf.reshape(target_cos_sim, target_cos_sim.shape[0]*target_cos_sim.shape[1])
                score, index = tf.math.top_k(flattened, k=1)
                tiled_set_label = tf.tile(set_label, [1, item_label.shape[-1]])
                pred_item_label = tf.gather_nd(item_label, indices[tf.unravel_index(index, target_cos_sim.shape)[0][0].numpy()])
                pred_set_label = tf.gather_nd(tiled_set_label, indices[tf.unravel_index(index, target_cos_sim.shape)[0][0].numpy()])
                pred_set_label = tf.cast(pred_set_label, tf.int64)


            pred_id_data.append(tf.stack([pred_set_label, pred_item_label],axis=0))
        pred_id_data_batch.append(pred_id_data)
        pred_id_data = []

    return tf.stack(pred_id_data_batch)
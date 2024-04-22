import argparse
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import os


#----------------------------
# model names
def mode_name(mode):
    mode_name = ['maxPooling','poolingMA','CSS','setRepVec_biPMA','setRepVec_pivot']

    return mode_name[mode]
#----------------------------

#----------------------------
# set func names
def set_func_name(set_func):
    set_func_name = ['CS','BERTscore']

    return set_func_name[set_func]
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
    parser.add_argument('-set_func', type=int, default=0, help='how to evaluate set similarity, CS:0, BERTscore:1, default=0')

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
    parser.add_argument('-set_func', type=int, default=0, help='how to evaluate set similarity, CS:0, BERTscore:1, default=0')
    return parser
#----------------------------

#----------------------------
# plot images in specified sets
def plotImg(imgs,set_IDs,msg="",fname="img_in_sets"):
    _, n_item, _, _, _ = imgs.shape
    n_set = len(set_IDs)
    #fig = plt.figure(figsize=(20,5))
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

    path = os.path.join(path,"result/loss_acc.png")
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

#loss function for Set Retrieval task
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
        overlapping_loss = 0
        '''
        if sort_indices[0] == sort_indices[1] == sort_indices[2]:
            overlapping_loss += (abs(sort_score[0]-sort_score[1])+abs(sort_score[1]-sort_score[2]))
        elif sort_indices[0] == sort_indices[2]:
            overlapping_loss += abs(sort_score[0]-sort_score[1])
        elif sort_indices[2] == sort_indices[1]:
            overlapping_loss += abs(sort_score[2]-sort_score[1])
        elif sort_indices[0] == sort_indices[1]:
            overlapping_loss += abs(sort_score[0]-sort_score[1])
        '''

    Loss = sum(hingelosssum)/len(hingelosssum) #+(overlapping_loss/len(simMaps)) 
    
    return Loss
#----------------------------

def f1_bert_score(simMaps, score):
    
    """アイテム集合と予測ベクトルとのbertスコアを評価する関数"""
    """対角成分のスコアが上位10%にあれば1 そうでなければ0"""
    threk = int(len(score)*0.1)
    
    accuracy = np.zeros((len(score), 1))
    for batch_ind in range(len(score)):
        f1_score = score[batch_ind]
        sort_score, sort_index = tf.math.top_k(f1_score, k=threk)
        if batch_ind in sort_index:
            accuracy[batch_ind] += 1

    return accuracy
    #return tf.linalg.trace(score)

#----------------------------
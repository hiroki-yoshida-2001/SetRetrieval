import tensorflow as tf
import pickle
import glob
import numpy as np
import pdb
import os
import sys

#-------------------------------
class trainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, year=2017, split=0, batch_size=20, max_item_num=5, max_data=np.inf):
        data_path = f"pickle_data/{year}-{year}-split{split}"
        self.max_item_num = max_item_num
        self.batch_size = batch_size
        
        # load train data
        with open(f'{data_path}/train.pkl', 'rb') as fp:
            self.x_train = pickle.load(fp)
            self.y_train = pickle.load(fp)
        self.train_num = len(self.x_train)

        # limit data
        if self.train_num > max_data:
            self.train_num = max_data

        # load validation data
        with open(f'{data_path}/valid.pkl', 'rb') as fp:
            self.x_valid = pickle.load(fp)
            self.y_valid = pickle.load(fp)
        self.valid_num = len(self.x_valid)        

        # width and height of image
        self.dim = len(self.x_train[0][0])

        # shuffle index
        self.inds = np.arange(len(self.x_train))
        self.inds_shuffle = np.random.permutation(self.inds)

    def __getitem__(self, index):
        x, x_size, y = self.data_generation(self.x_train, self.y_train, self.inds_shuffle, index)
        return (x, x_size), y

    def data_generation(self, x, y, inds, index):
        
        if index >= 0:
            # extract x and y
            start_ind = index * self.batch_size
            batch_inds = inds[start_ind:start_ind+self.batch_size]
            x_tmp = [x[i] for i in batch_inds]
            y_tmp = [y[i] for i in batch_inds]
            batch_size = self.batch_size
        else:
            x_tmp = x
            y_tmp = y
            batch_size = len(x_tmp)

        # split x
        x_batch = []
        x_size_batch = []
        y_batch =[]
        split_num = 2
        for ind in range(batch_size):
            x_tmp_split = np.array_split(x_tmp[ind][np.random.permutation(len(x_tmp[ind]))],split_num)
            x_tmp_split_pad = [np.vstack([x, np.zeros([np.max([0,self.max_item_num-len(x)]),self.dim])])[:self.max_item_num] for x in x_tmp_split] # zero padding

            x_batch.append(x_tmp_split_pad)

            # x_size is adjusted with max item number if it's over max item number.  
            if (len(x_tmp_split[0]) <= self.max_item_num) and (len(x_tmp_split[1]) <= self.max_item_num):
                x_size_batch.append([len(x_tmp_split[i]) for i in range(split_num)])
            elif (len(x_tmp_split[0]) > self.max_item_num) and (len(x_tmp_split[1]) <= self.max_item_num):
                x_size_batch.append([self.max_item_num, len(x_tmp_split[1])])
            elif (len(x_tmp_split[1]) > self.max_item_num) and (len(x_tmp_split[0]) <= self.max_item_num):
                x_size_batch.append([len(x_tmp_split[1]), self.max_item_num])
            else:
                x_size_batch.append([self.max_item_num, self.max_item_num])
            
            y_batch.append(np.ones(split_num)*y_tmp[ind])


        x_batch = np.vstack(x_batch)
        x_size_batch = np.hstack(x_size_batch).astype(np.float32)
        y_batch = np.hstack(y_batch)

        return x_batch, x_size_batch, y_batch

    def data_generation_val(self):
        
        x_valid, x_size_val, y_valid = self.data_generation(self.x_valid, self.y_valid, self.inds, -1)
        return x_valid, x_size_val, y_valid

    def __len__(self):
        # number of batches in one epoch
        batch_num = int(self.train_num/self.batch_size)

        return batch_num

    def on_epoch_end(self):
        # shuffle index
        self.inds_shuffle = np.random.permutation(self.inds)
#-------------------------------

#-------------------------------
class testDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, year=2017, split=0, cand_num=4):
        self.data_path = f"pickle_data/{year}-{year}-split{split}"
        self.cand_num = cand_num
        # (number of groups in one batch) = (cand_num) + (one query)
        self.batch_grp_num = cand_num + 1

        # load data
        with open(f'{self.data_path}/test_example_cand{self.cand_num}.pkl', 'rb') as fp:
            self.x = pickle.load(fp)
            self.x_size = pickle.load(fp)
            self.y = pickle.load(fp)
#-------------------------------
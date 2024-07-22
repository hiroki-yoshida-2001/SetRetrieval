import tensorflow as tf
import pickle
import glob
import numpy as np
import pdb
import os
import sys

#-------------------------------
class trainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, year=2017, split=0, batch_size=20, max_item_num=5, max_data=np.inf, mlp_flag=False):
        data_path =  "/data2/yoshida/mastermatching/data/forpack/pickles/2017-2017-split0" #f"pickle_data/{year}-{year}-split{split}"
        self.max_item_num = max_item_num
        self.batch_size = batch_size
        self.isMLP = mlp_flag
        
        # load train data
        #pdb.set_trace()
        with open(f'{data_path}/train.pkl', 'rb') as fp:
            self.x_train = pickle.load(fp)
            self.y_train = pickle.load(fp)

            self.category1_train = pickle.load(fp)
            self.category2_train = pickle.load(fp)
            self.item_label_train = pickle.load(fp)

        self.train_num = len(self.x_train)

        # limit data
        if self.train_num > max_data:
            self.train_num = max_data

        # load validation data
        with open(f'{data_path}/valid.pkl', 'rb') as fp:
            self.x_valid = pickle.load(fp)
            self.y_valid = pickle.load(fp)

            self.category1_valid = pickle.load(fp)
            self.category2_valid = pickle.load(fp)
            self.item_label_valid = pickle.load(fp)
        self.valid_num = len(self.x_valid)  

        # load test data
        with open(f'{data_path}/test.pkl', 'rb') as fp:
            self.x_test = pickle.load(fp)
            self.y_test = pickle.load(fp)

            self.category1_test = pickle.load(fp)
            self.category2_test = pickle.load(fp)
            self.item_label_test = pickle.load(fp)
        self.test_num = len(self.x_test)        

        # width and height of image
        self.dim = len(self.x_train[0][0])

        # shuffle index
        self.inds = np.arange(len(self.x_train))
        self.inds_shuffle = np.random.permutation(self.inds)

        # data for pretrain task
        self.x_pretrain = np.concatenate(self.x_train, axis=0)
        self.y_pretrain = np.concatenate(self.category2_train, axis=0)

        # label encoding (only for train data) generatorで毎回呼ぶと時間がかかるため
        unique_labels, counts = np.unique(self.y_pretrain, return_counts=True)
        self.label_to_index = {label: index for index, label in enumerate(unique_labels)}

        self.y_pretrain = np.array([self.label_to_index[label] for label in self.y_pretrain])
        self.inds_pr = np.arange(len(self.y_pretrain))
        self.inds_pr_shuffle = np.random.permutation(self.inds_pr)


        


    def __getitem__(self, index):
        if self.isMLP: 
            x, y = self.pretrain_data_generation(self.x_pretrain, self.y_pretrain, self.inds_pr_shuffle, index)
            return x, y
        else: #集合としてのバッチ学習
            x, x_size, y = self.data_generation(self.x_train, self.y_train, self.inds_shuffle, index)
            return (x, x_size), y
    
    def pretrain_data_generation(self, x, y, inds, index):

        if index >= 0:
            start_idx = index * self.batch_size
            end_idx = min((index + 1) * self.batch_size, len(y))
            excerpt = inds[start_idx:end_idx]

            return x[excerpt], y[excerpt]
        else:
            y = np.array([self.label_to_index[label] for label in y])
            return x, y

    def data_generation(self, x, y, inds, index, category_1=0, category_2=0, item_label=0):
        #pdb.set_trace()
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

            if not category_1 == 0:
                #pdb.set_trace()
                category_1_tmp = category_1
                category_2_tmp = category_2
                item_label_tmp = item_label

        # split x
        x_batch = []
        x_size_batch = []
        y_batch =[]
        split_num = 2
        if not category_1 == 0:
            #pdb.set_trace()
            category_1_batch = []
            category_2_batch = []
            item_label_batch = []

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

            if not category_1 == 0:
                category_1_tmp = [np.array(sublist, dtype=int) for sublist in category_1_tmp]
                category_1 = [np.array(sublist, dtype=int) for sublist in category_1]
                category_1_split = np.array_split(category_1_tmp[ind][np.random.permutation(len(category_1_tmp[ind]))],split_num)
                category_1_split_pad = []
                #category_1_split_pad = [arr if len(arr) >= self.max_item_num else np.pad(arr, (0, self.max_item_num - len(arr)), mode='constant') for arr in category_1_split]
                for arr in category_1_split:
                    if len(arr) > self.max_item_num:
                        arr = arr[:self.max_item_num]
                    elif len(arr) < self.max_item_num:
                        arr = np.pad(arr, (0, self.max_item_num - len(arr)), mode='constant')
                    category_1_split_pad.append(arr)
                category_1_batch.append(category_1_split_pad)

                category_2_tmp = [np.array(sublist, dtype=int) for sublist in category_2_tmp]
                category_2 = [np.array(sublist, dtype=int) for sublist in category_2]
                category_2_split = np.array_split(category_2_tmp[ind][np.random.permutation(len(category_2_tmp[ind]))],split_num)
                
                category_2_split_pad = []
                for arr in category_2_split:
                    if len(arr) > self.max_item_num:
                        arr = arr[:self.max_item_num]
                    elif len(arr) < self.max_item_num:
                        arr = np.pad(arr, (0, self.max_item_num - len(arr)), mode='constant')
                    category_2_split_pad.append(arr)
                category_2_batch.append(category_2_split_pad)

                item_label_tmp = [np.array(sublist, dtype=int) for sublist in item_label]
                item_label = [np.array(sublist, dtype=int) for sublist in item_label]
                item_label_split = np.array_split(item_label_tmp[ind][np.random.permutation(len(item_label_tmp[ind]))],split_num)
                
                item_label_pad = []
                for arr in item_label_split:
                    if len(arr) > self.max_item_num:
                        arr = arr[:self.max_item_num]
                    elif len(arr) < self.max_item_num:
                        arr = np.pad(arr, (0, self.max_item_num - len(arr)), mode='constant')
                    item_label_pad.append(arr)
                item_label_batch.append(item_label_pad)
                # item_split = 


        x_batch = np.vstack(x_batch)
        x_size_batch = np.hstack(x_size_batch).astype(np.float32)
        y_batch = np.hstack(y_batch)
        if not category_1 == 0:
            #pdb.set_trace()
            category_1_batch = np.vstack(category_1_batch)
            category_2_batch = np.vstack(category_2_batch)
            item_label_batch = np.vstack(item_label_batch)
            
        if not category_1 == 0:
            return x_batch, x_size_batch, y_batch, category_1_batch, category_2_batch, item_label_batch
        return x_batch, x_size_batch, y_batch

    def data_generation_val(self):
        if self.isMLP:
            x_valid, y_valid = self.pretrain_data_generation(np.concatenate(self.x_valid, axis=0), np.concatenate(self.category2_valid, axis=0), self.inds, -1)
            return x_valid, y_valid
        else:
            x_valid, x_size_val, y_valid = self.data_generation(self.x_valid, self.y_valid, self.inds, -1)
            return x_valid, x_size_val, y_valid
    def data_generation_train(self):
        if self.isMLP:
            x_train, y_train = self.pretrain_data_generation(np.concatenate(self.x_train, axis=0), np.concatenate(self.category2_train, axis=0), self.inds, -1)
            return x_train, y_train
        else:
            x_train, x_size_train, y_train = self.data_generation(self.x_train, self.y_train, self.inds, -1)
            return x_train, x_size_train, y_train
    
    def data_generation_test(self):
        
        x_test, x_size_test, y_test, category1_test, category2_test, item_label_test = self.data_generation(self.x_test, self.y_test, self.inds,  -1, category_1=self.category1_test, category_2=self.category2_test, item_label=self.item_label_test)
        return x_test, x_size_test, y_test, category1_test, category2_test, item_label_test

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
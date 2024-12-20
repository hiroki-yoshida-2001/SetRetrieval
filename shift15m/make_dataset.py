import tensorflow as tf
import pickle
import glob
import numpy as np
import pdb
import os
import sys
from collections import Counter
import random

#-------------------------------
class trainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, year=2017, split=0, batch_size=20, max_item_num=5, max_data=np.inf, mlp_flag=False, use_all_pred=True):
        data_path =  "/data2/yoshida/mastermatching/data/forpack/pickles/2017-2017-split0" #f"pickle_data/{year}-{year}-split{split}"
        self.max_item_num = max_item_num
        self.batch_size = batch_size
        self.isMLP = mlp_flag
        self.use_all_pred = use_all_pred

        # load train data
        
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
        self.y_pretrain_c1 = np.concatenate(self.category1_train, axis=0)

        # label encoding (only for train data) generatorで毎回呼ぶと時間がかかるため
        unique_labels, counts = np.unique(self.y_pretrain, return_counts=True)
        self.label_to_index = {label: index for index, label in enumerate(unique_labels)}

        # c1 label encoding 
        c1_unique_labels, c1_counts = np.unique(self.y_pretrain_c1, return_counts=True)
        self.c1_label_to_index = {label: index for index, label in enumerate(c1_unique_labels)}

        self.y_pretrain = np.array([self.label_to_index[label] for label in self.y_pretrain])
        self.inds_pr = np.arange(len(self.y_pretrain))
        self.inds_pr_shuffle = np.random.permutation(self.inds_pr)

        # self.category2_train_int = [[int(num) for num in sublist] for sublist in self.category2_train]
        self.category2_train_int = [[self.label_to_index[num] for num in sublist] for sublist in self.category2_train]
        # self.category2_valid_int = [[int(num) for num in sublist] for sublist in self.category2_valid]
        self.category2_valid_int = [[self.label_to_index[num] for num in sublist] for sublist in self.category2_valid]
        # self.category2_test_int = [[int(num) for num in sublist] for sublist in self.category2_test]
        self.category2_test_int = [[self.label_to_index[num] for num in sublist] for sublist in self.category2_test]
        self.category1_test_int = [[self.c1_label_to_index[num] for num in sublist] for sublist in self.category1_test]



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
                category_1_tmp = category_1
                category_2_tmp = category_2
                item_label_tmp = item_label
                c2_int_tmp = self.category1_test_int

        # split x
        x_batch = []
        x_size_batch = []
        y_batch =[]
        split_num = 2
        if not category_1 == 0:
            category_1_batch = []
            category_2_batch = []
            item_label_batch = []
            c2_int_batch = []

        for ind in range(batch_size):
            random_indices = np.random.permutation(len(x_tmp[ind]))
            x_tmp_split = np.array_split(x_tmp[ind][random_indices],split_num)
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
                category_1_split = np.array_split(category_1_tmp[ind][random_indices],split_num)
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
                category_2_split = np.array_split(category_2_tmp[ind][random_indices],split_num)
                
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
                item_label_split = np.array_split(item_label_tmp[ind][random_indices],split_num)
                
                item_label_pad = []
                for arr in item_label_split:
                    if len(arr) > self.max_item_num:
                        arr = arr[:self.max_item_num]
                    elif len(arr) < self.max_item_num:
                        arr = np.pad(arr, (0, self.max_item_num - len(arr)), mode='constant')
                    item_label_pad.append(arr)
                item_label_batch.append(item_label_pad)
                # item_split = 

                c2_tmp = [np.array(sublist, dtype=int) for sublist in c2_int_tmp]
                c2 = [np.array(sublist, dtype=int) for sublist in c2_int_tmp]
                c2_split = np.array_split(c2_tmp[ind][random_indices],split_num)
                
                c2_split_pad = []
                for arr in c2_split:
                    if len(arr) > self.max_item_num:
                        arr = arr[:self.max_item_num]
                    elif len(arr) < self.max_item_num:
                        arr = np.pad(arr, (0, self.max_item_num - len(arr)), mode='constant', constant_values=len(self.c1_label_to_index))
                    c2_split_pad.append(arr)
                c2_int_batch.append(c2_split_pad)




        x_batch = np.vstack(x_batch)
        x_size_batch = np.hstack(x_size_batch).astype(np.float32)
        y_batch = np.hstack(y_batch)
        if not category_1 == 0:
            category_1_batch = np.vstack(category_1_batch)
            category_2_batch = np.vstack(category_2_batch)
            item_label_batch = np.vstack(item_label_batch)
            c2_int_batch = np.vstack(c2_int_batch)
            
        if not category_1 == 0:
            return x_batch, x_size_batch, y_batch, category_1_batch, category_2_batch, item_label_batch, c2_int_batch
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
        
        x_test, x_size_test, y_test, category1_test, category2_test, item_label_test, c2_test = self.data_generation(self.x_test, self.y_test, self.inds,  -1, category_1=self.category1_test, category_2=self.category2_test, item_label=self.item_label_test)
        return x_test, x_size_test, y_test, category1_test, category2_test, item_label_test, c2_test

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


# train_generator (written with tf.data)
class DataGenerator:
    def  __init__(self, year=2017, split=0, batch_size=20, max_item_num=5, max_data=np.inf, mlp_flag=False, use_all_pred=False):
        data_path =  "/data2/yoshida/mastermatching/data/forpack/pickles/2017-2017-split0" #f"pickle_data/{year}-{year}-split{split}"
        self.max_item_num = max_item_num
        self.batch_size = batch_size
        self.isMLP = mlp_flag
        self.use_all_pred = use_all_pred
        
        # load train data
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

        self.inds_vr = np.arange(len(self.x_valid))

        # data for pretrain task
        self.x_pretrain = np.concatenate(self.x_train, axis=0)
        self.y_pretrain = np.concatenate(self.category2_train, axis=0)
        self.y_pretrain_c1 = np.concatenate(self.category1_train, axis=0)

        # c2 label encoding (only for train data) generatorで毎回呼ぶと時間がかかるため
        unique_labels, counts = np.unique(self.y_pretrain, return_counts=True)
        self.label_to_index = {label: index for index, label in enumerate(unique_labels)}

        # c1 label encoding 
        c1_unique_labels, c1_counts = np.unique(self.y_pretrain_c1, return_counts=True)
        self.c1_label_to_index = {label: index for index, label in enumerate(c1_unique_labels)}

        # self.category2_train_int = [[int(num) for num in sublist] for sublist in self.category2_train]
        self.category2_train_int = [[self.label_to_index[num] for num in sublist] for sublist in self.category2_train]
        # self.category2_valid_int = [[int(num) for num in sublist] for sublist in self.category2_valid]
        self.category2_valid_int = [[self.label_to_index[num] for num in sublist] for sublist in self.category2_valid]
        
        self.category1_train_int = [[self.c1_label_to_index[num] for num in sublist] for sublist in self.category1_train]
        self.category1_valid_int = [[self.c1_label_to_index[num] for num in sublist] for sublist in self.category1_valid]
        
        self.y_pretrain = np.array([self.label_to_index[label] for label in self.y_pretrain])
        self.inds_pr = np.arange(len(self.y_pretrain))
        self.inds_pr_shuffle = np.random.permutation(self.inds_pr)
        

    # -------------------------------------
    # 各要素の割合を計算, バッチ内データ加工のための関数
    def calculate_ratios(self, array_list):
        # 配列リストを1次元に結合してユニークな要素とそのカウントを取得
        unique_elements, counts = np.unique(np.concatenate(array_list), return_counts=True)
        
        # カウントに基づいて割合を計算
        total_count = np.sum(counts)
        target_ratios = {elem: count / total_count for elem, count in zip(unique_elements, counts)}
        
        return target_ratios
    # -------------------------------------

    # シャッフル後の0番目の配列が元の配列リストで何番目だったかを調べる関数
    def find_original_index(self, shuffled_array, original_array_list):
        for i, original_array in enumerate(original_array_list):
            if np.array_equal(shuffled_array, original_array):
                return i
        return None  # 見つからなかった場合

    # -------------------------------------
    # 指定された割合に基づいて要素のインデックスを抽出する関数, バッチ内データ加工のための関数
    def extract_indices_based_on_ratios(self, array_list, target_ratios, total_sample_size):
        # インデックスを選択するためのリスト
        original_array = array_list
        shuffled_list = array_list.copy()

        selected_indices = []
        # まず、ターゲットの割合に従って選択
        for elem, ratio in target_ratios.items():
            num_to_select = int(total_sample_size * ratio)  # 割合に応じて選ぶ数を計算
            random.shuffle(shuffled_list)
            # 各配列内で指定された値を持つ要素のインデックスを見つける
            for i, arr in enumerate(shuffled_list):
                # その配列に指定の要素が含まれているか確認
                elem_indices = np.where(arr == elem)[0]
                if len(elem_indices) > 0:
                    # 抽出するインデックスを決定
                    index = self.find_original_index(arr, original_array)
                    selected_indices.append(index)
                    # 必要な数に達した場合
                    num_to_select -= 1
                    if num_to_select <= 0:
                        break
        
        # 残りのインデックスをランダムに選択
        remaining_count = total_sample_size - len(selected_indices)
        
        if remaining_count > 0:
            # 残りのインデックスをランダムに選ぶ
            remaining_indices = [i for i, arr in enumerate(shuffled_list)
                                if i not in selected_indices]
            selected_indices.append(random.sample(remaining_indices, remaining_count))
        
        return selected_indices
    # -------------------------------------

    def train_generator(self):
        np.random.shuffle(self.inds)
        
        for index in range(0, len(self.inds), self.batch_size):
            start_ind = index

            batch_inds = self.inds[start_ind:start_ind + self.batch_size]
            x_tmp = [self.x_train[i] for i in batch_inds]
            y_tmp = [self.y_train[i] for i in batch_inds]
            if not self.use_all_pred:
                category2_tmp = [np.array(self.category2_train_int[i]) for i in batch_inds]
                category1_tmp = [np.array(self.category1_train_int[i]) for i in batch_inds]
            # xとyをスプリットし、パディングを適用
            x_batch = []
            x_size_batch = []
            y_batch = []
            c_batch = []
            c1_batch = []
            for ind in range(len(x_tmp)):
                random_indices = np.random.permutation(len(x_tmp[ind]))
                x_tmp_split = np.array_split(x_tmp[ind][random_indices], 2)
                x_tmp_split_pad = [
                    np.pad(
                        x[:self.max_item_num],  # xの長さがmax_item_numを超える場合は切り捨て
                        ((0, max(0, self.max_item_num - len(x))), (0, 0)),  # パディングを適用
                        mode='constant'
                    ) for x in x_tmp_split
                ]
                if not self.use_all_pred:
                    c_tmp_split = np.array_split(category2_tmp[ind][random_indices], 2)
                    c_tmp_split_pad = [
                        np.pad(
                            c[:self.max_item_num],  # xの長さがmax_item_numを超える場合は切り捨て
                            ((0, max(0, self.max_item_num - len(c))),),  # -1ラベルを設定(検索ベクトルで無視)
                            mode='constant', constant_values=len(self.label_to_index)
                        ) for c in c_tmp_split
                    ]

                    c1_tmp_split = np.array_split(category1_tmp[ind][random_indices], 2)
                    c1_tmp_split_pad = [
                        np.pad(
                            c[:self.max_item_num],  # xの長さがmax_item_numを超える場合は切り捨て
                            ((0, max(0, self.max_item_num - len(c))),),  # -1ラベルを設定(検索ベクトルで無視)
                            mode='constant', constant_values=len(self.label_to_index)
                        ) for c in c1_tmp_split
                    ]
                x_batch.append(x_tmp_split_pad)

                x_size_batch.append([min(len(split), self.max_item_num) for split in x_tmp_split])
                y_batch.append(np.ones(2) * y_tmp[ind])
                if not self.use_all_pred:
                    c_batch.append(c_tmp_split_pad)
                    c1_batch.append(c1_tmp_split_pad)
            
            x_batch = np.vstack(x_batch)
            x_size_batch = np.hstack(x_size_batch)
            y_batch = np.hstack(y_batch)

            if not self.use_all_pred:
                c_batch = np.vstack(c_batch)
                c1_batch = np.vstack(c1_batch)

            
            if not self.use_all_pred:
                yield (x_batch, x_size_batch, c_batch, c1_batch), y_batch
            else:
                yield (x_batch, x_size_batch), y_batch
    def validation_generator(self):
        for index in range(0, len(self.inds_vr), self.batch_size):
            start_ind = index

            batch_inds = self.inds_vr[start_ind:start_ind + self.batch_size]
            # batch_inds = self.inds_vr[start_ind:start_ind + self.batch_size]
            x_tmp = [self.x_valid[i] for i in batch_inds]
            y_tmp = [self.y_valid[i] for i in batch_inds]

            if not self.use_all_pred:
                category_tmp = [np.array(self.category2_valid_int[i]) for i in batch_inds]
                category1_tmp = [np.array(self.category1_valid_int[i]) for i in batch_inds]

            # xとyをスプリットし、パディングを適用
            x_batch = []
            x_size_batch = []
            y_batch = []
            c_batch = []
            c1_batch = []
            for ind in range(len(x_tmp)):
                x_tmp_split = np.array_split(x_tmp[ind], 2)
                x_tmp_split_pad = [
                    np.pad(
                        x[:self.max_item_num],  # xの長さがmax_item_numを超える場合は切り捨て
                        ((0, max(0, self.max_item_num - len(x))), (0, 0)),  # パディングを適用
                        mode='constant'
                    ) for x in x_tmp_split
                ]

                if not self.use_all_pred:
                    c_tmp_split = np.array_split(category_tmp[ind], 2)
                    c_tmp_split_pad = [
                        np.pad(
                            c[:self.max_item_num],  # xの長さがmax_item_numを超える場合は切り捨て
                            ((0, max(0, self.max_item_num - len(c))),),  # -1ラベルを付与(検索ベクトルで無視)
                            mode='constant', constant_values=len(self.label_to_index)
                        ) for c in c_tmp_split
                    ]
                    c1_tmp_split = np.array_split(category1_tmp[ind], 2)
                    c1_tmp_split_pad = [
                        np.pad(
                            c[:self.max_item_num],  # xの長さがmax_item_numを超える場合は切り捨て
                            ((0, max(0, self.max_item_num - len(c))),),  # -1ラベルを付与(検索ベクトルで無視)
                            mode='constant', constant_values=len(self.label_to_index)
                        ) for c in c1_tmp_split
                    ]
                x_batch.append(x_tmp_split_pad)

                x_size_batch.append([min(len(split), self.max_item_num) for split in x_tmp_split])
                y_batch.append(np.ones(2) * y_tmp[ind])

                if not self.use_all_pred:
                    c_batch.append(c_tmp_split_pad)
                    c1_batch.append(c1_tmp_split_pad)
            
            x_batch = np.vstack(x_batch)
            x_size_batch = np.hstack(x_size_batch)
            y_batch = np.hstack(y_batch)
            if not self.use_all_pred:
                c_batch = np.vstack(c_batch)
                c1_batch = np.vstack(c1_batch)

            
            if not self.use_all_pred:
                yield (x_batch, x_size_batch, c_batch, c1_batch), y_batch
            else:
                yield (x_batch, x_size_batch), y_batch

            # yield (x_batch, x_size_batch), y_batch

    def get_train_dataset(self):
        if self.use_all_pred:
            return tf.data.Dataset.from_generator(
                self.train_generator,
                    output_types=((tf.float64, tf.float32), tf.float64),
                    output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,])), tf.TensorShape([None,]))
                    
            )
        else:
            return tf.data.Dataset.from_generator(
                self.train_generator,
                    output_types=((tf.float64, tf.float32, tf.int64, tf.int64), tf.float64),
                    output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,]), tf.TensorShape([None, self.max_item_num]), tf.TensorShape([None, self.max_item_num])), tf.TensorShape([None,]))
                    
            )
        
    
    def get_validation_dataset(self):     
        if self.use_all_pred:
            return tf.data.Dataset.from_generator(
                self.validation_generator,
                output_types=((tf.float64, tf.float32), tf.float64),
                output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,])), tf.TensorShape([None,]))
                    
            )
        else:
            return tf.data.Dataset.from_generator(
                self.validation_generator,
                output_types=((tf.float64, tf.float32, tf.int64, tf.int64), tf.float64),
                output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,]), tf.TensorShape([None, self.max_item_num]), tf.TensorShape([None, self.max_item_num])), tf.TensorShape([None,]))
                    
            )
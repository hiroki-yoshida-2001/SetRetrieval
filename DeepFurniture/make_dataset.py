import tensorflow as tf
import pickle
import glob
import numpy as np
import pdb
import os
import sys
from collections import Counter
import random

def interleave_arrays(A, B):
    """
    AとBを交互に結合する。
    A[0], B[0], A[1], B[1], ..., A[-1], B[-1]という順に並ぶようにする。

    Args:
        A: NumPy配列 (shape: [N, ..., ...])
        B: NumPy配列 (shape: [N, ..., ...])

    Returns:
        interleaved: 交互に結合されたNumPy配列 (shape: [2*N, ..., ...])
    """
    # AとBの0軸(先頭次元)を拡張して結合
    interleaved = np.empty((A.shape[0] + B.shape[0], *A.shape[1:]), dtype=A.dtype)
    interleaved[0::2] = A  # 偶数番目にAを配置
    interleaved[1::2] = B  # 奇数番目にBを配置
    return interleaved
#-------------------------------

# train_generator (written with tf.data)
class DataGenerator:
    def  __init__(self, year=2017, split=0, batch_size=20, max_item_num=5, max_data=np.inf, mlp_flag=False, set_loss=False, whitening_path = None, seed_path = None):
        data_path =  "/data2/yoshida/mastermatching/DeepFurniture" # train, valid, test.pickleのあるフォルダパス
        self.max_item_num = max_item_num
        self.batch_size = batch_size
        self.isMLP = mlp_flag
        self.set_loss = set_loss
        
        # load train data
        with open(f'{data_path}/train_fur.pkl', 'rb') as fp:
            self.query_tr, self.positive_tr, self.y_tr, self.y_catQ_tr, self.y_catP_tr, self.x_size_tr, self.y_size_tr = pickle.load(fp)
        self.x_train = interleave_arrays(self.query_tr, self.positive_tr)
        self.label_tr = interleave_arrays(self.y_catQ_tr, self.y_catP_tr)
        self.x_size_train = interleave_arrays(self.x_size_tr, self.y_size_tr)
        # shift15mと同じラベルの付け方に変更
        self.c_label_tr = self.label_tr -1
        self.max_label =  self.c_label_tr.max()
        self.c_label_tr[self.c_label_tr==-1] = self.max_label + 1
        # self.y_train = np.repeat(self.y_tr, 2)

        self.train_num = len(self.x_train)
        # feature vector dimension
        self.dim = len(self.query_tr[0][0])
        # pdb.set_trace()
        train_set_size, _ , _ = self.x_train.shape
        seed_origin = np.reshape(self.x_train,[train_set_size*self.max_item_num, self.dim])
        label_origin = np.reshape(self.c_label_tr, [train_set_size*self.max_item_num])
        gause_noise = np.random.randn(self.dim, 256)
        whitening_seed = np.matmul(seed_origin, gause_noise)
        
        if whitening_path:
            whitening_mean = np.mean(whitening_seed, axis=0)
            whitening_std = np.std(whitening_seed, axis=0)

            with open("furniture_gausenoise.pkl", "wb") as fp:
                pickle.dump(gause_noise, fp)
                pickle.dump(whitening_mean, fp)
                pickle.dump(whitening_std, fp)

        if seed_path:
            seed_vec_list = []
            for i in range(self.max_label+1):
                seed_vec = np.mean(seed_origin[np.where(label_origin==i)],axis=0)
                seed_vec_list.append(seed_vec)
            seed_vec_list = np.stack(seed_vec_list)
            with open("furniture_seed.pkl", "wb") as fp:
                pickle.dump(seed_vec_list, fp)
        
        # limit data
        if self.train_num > max_data:
            self.train_num = max_data

        # load validation data
        self.y_pretrain = np.reshape(self.c_label_tr, -1)
        
        # load validation data
        with open(f'{data_path}/validation_fur.pkl', 'rb') as fp:
            self.query_val, self.positive_val, self.y_val, self.y_catQ_val, self.y_catP_val, self.x_size_val, self.y_size_val = pickle.load(fp)

        self.x_valid = interleave_arrays(self.query_val, self.positive_val)
        self.x_size_valid = interleave_arrays(self.x_size_val, self.y_size_val)
        self.val_set_size, self.val_item_size, _ = self.x_valid.shape
        self.label_val = interleave_arrays(self.y_catQ_val, self.y_catP_val)
        # shift15mと同じラベルの付け方に変更
        self.c_label_val = self.label_val -1
        # self.max_label =  self.c_label_val.max()
        self.c_label_val[self.c_label_val==-1] = self.max_label + 1

        self.y_prevalid = np.reshape(self.c_label_val, -1)

         # load test data
        with open(f'{data_path}/test_fur.pkl', 'rb') as fp:
            self.query_test, self.positive_test, self.y_test, self.y_catQ_test, self.y_catP_test, self.x_size_test, self.y_size_test, self.x_id_test, self.y_id_test, self.scene_id_dict = pickle.load(fp) 
        
        # shuffle index
        self.inds = np.arange(len(self.x_train))
        self.inds_shuffle = np.random.permutation(self.inds)

        self.inds_vr = np.arange(len(self.x_valid))

        self.x_test = interleave_arrays(self.query_test, self.positive_test)
        self.x_size_test = interleave_arrays(self.x_size_test, self.y_size_test)
        self.x_id_test = interleave_arrays(self.x_id_test, self.y_id_test)
        self.test_set_size, self.test_item_size, _ = self.x_test.shape
        self.label_test = interleave_arrays(self.y_catQ_test, self.y_catP_test)
        self.y_test = np.repeat(self.y_test, 2)
        # shift15mと同じラベルの付け方に変更
        self.c_label_test = self.label_test -1
        # self.max_label =  self.c_label_val.max()
        self.c_label_test[self.c_label_test==-1] = self.max_label + 1
        self.inds_test = np.arange(len(self.x_test))
        # data for pretrain task
        # self.x_pretrain = np.concatenate(self.x_train, axis=0)
        # self.y_pretrain = np.concatenate(self.category2_train, axis=0)
        # self.y_pretrain_c1 = np.concatenate(self.category1_train, axis=0)
        
        # # negative gallery for validation
        # self.x_prevalid = np.concatenate(self.x_valid, axis=0)
        # self.y_prevalid = np.concatenate(self.category2_valid, axis=0)

        # # c2 label encoding (only for train data) generatorで毎回呼ぶと時間がかかるため
        # unique_labels, counts = np.unique(self.y_pretrain, return_counts=True)
        # self.label_to_index = {label: index for index, label in enumerate(unique_labels)}

        # # c1 label encoding 
        # c1_unique_labels, c1_counts = np.unique(self.y_pretrain_c1, return_counts=True)
        # self.c1_label_to_index = {label: index for index, label in enumerate(c1_unique_labels)}

        # # self.category2_train_int = [[int(num) for num in sublist] for sublist in self.category2_train]
        # self.category2_train_int = [[self.label_to_index[num] for num in sublist] for sublist in self.category2_train]
        # # self.category2_valid_int = [[int(num) for num in sublist] for sublist in self.category2_valid]
        # self.category2_valid_int = [[self.label_to_index[num] for num in sublist] for sublist in self.category2_valid]
        
        # self.category1_train_int = [[self.c1_label_to_index[num] for num in sublist] for sublist in self.category1_train]
        # self.category1_valid_int = [[self.c1_label_to_index[num] for num in sublist] for sublist in self.category1_valid]
        
        # self.y_pretrain = np.array([self.label_to_index[label] for label in self.y_pretrain])
        self.inds_pr = np.arange(len(self.y_pretrain))
        self.inds_pr_shuffle = np.random.permutation(self.inds_pr)
        # self.y_prevalid_dash = np.array([self.label_to_index[label] for label in self.y_prevalid])

        self.negative_item_num = 30
        self.negative_level = 1.00
        self.negative_level_val = 0.20
        
        # self.negative_item_num = 30
        # self.negative_level = 0.75
        
        self.class_dict = {} # self.x_pretrainに紐づくインデックスを格納する辞書
        self.classtoind_dict = {} # negative choosingで選ばれたインデックスからlabel_indicesのインデックスへ変換する際の辞書
        # cosine similarity calculation with Database

        self.inds_db = np.arange(0, self.x_train.shape[0]*self.x_train.shape[1]).reshape(self.x_train.shape[0], self.x_train.shape[1])
        self.inds_db_val = np.arange(0, self.x_valid.shape[0]*self.x_valid.shape[1]).reshape(self.x_valid.shape[0], self.x_valid.shape[1])
       
        self.gallery_set_size, self.gallery_item_size, self.dim = self.x_train.shape
        for label in range(self.max_label+1):  # ラベル0～self.negative_item_numに対して処理
            # 該当ラベルの要素のインデックスを取得
            label_indices = np.where(self.y_pretrain == label)[0]

            category_dx = np.reshape(self.x_train, [self.gallery_set_size*self.gallery_item_size, self.dim])[label_indices]
            l2_norms = np.linalg.norm(category_dx, axis=1, keepdims=True)  # Shape: (100, 1)

            # Normalize each vector
            normalized_vectors = category_dx / np.maximum(l2_norms, 1e-8)  # Prevent division by zero

            # Step 2: コサイン類似度計算
            # Compute the cosine similarity map using the dot product
            cosine_similarity_map = np.matmul(normalized_vectors, normalized_vectors.T)  # Shape: (100, 100)
            sorted_indices = np.argsort(cosine_similarity_map, axis=1)

            sorted_label = label_indices[sorted_indices]
            
            self.class_dict[label] = sorted_label
            self.classtoind_dict[label] = label_indices
        
        self.class_dict_val = {} # self.x_pretrainに紐づくインデックスを格納する辞書
        self.classtoind_dict_val = {} # negative choosingで選ばれたインデックスからlabel_indicesのインデックスへ変換する際の辞書
        for label in range(self.max_label+1):  # ラベル0～self.negative_item_numに対して処理
            # 該当ラベルの要素のインデックスを取得
            label_indices = np.where(self.y_prevalid == label)[0]
            
            category_dx = np.reshape(self.x_valid, [self.x_valid.shape[0]*self.x_valid.shape[1], self.dim])[label_indices]
            l2_norms = np.linalg.norm(category_dx, axis=1, keepdims=True)  # Shape: (100, 1)

            # Normalize each vector
            normalized_vectors = category_dx / np.maximum(l2_norms, 1e-8)  # Prevent division by zero

            # Step 2: コサイン類似度計算
            # Compute the cosine similarity map using the dot product
            cosine_similarity_map = np.matmul(normalized_vectors, normalized_vectors.T)  # Shape: (100, 100)
            sorted_indices = np.argsort(cosine_similarity_map, axis=1)

            sorted_label = label_indices[sorted_indices]
            
            self.class_dict_val[label] = sorted_label
            self.classtoind_dict_val[label] = label_indices
        # ------------------------------------------- 
        
        
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
        random_indices = np.random.permutation(len(self.inds))
        self.inds = self.inds[random_indices]
        x_train = self.x_train[random_indices]
        x_size_tr = self.x_size_train[random_indices]
        # y_tr = self.y_train[random_indices]
        c_label_tr = self.c_label_tr[random_indices]
        inds_db = self.inds_db[random_indices]
        # np.random.shuffle(self.inds)
        if not self.negative_level < 0.30:
            self.negative_level -= 0.01
        for index in range(0, len(self.inds), self.batch_size):
            start_ind = index
            
            batch_inds = self.inds[start_ind:start_ind + self.batch_size]
            
            # xとyをスプリットし、パディングを適用
            negative_batch = []
            x_batch = x_train[start_ind:start_ind + self.batch_size]
            # y_batch = y_tr[start_ind:start_ind + self.batch_size]
            y_batch = np.repeat(np.arange(int(len(x_batch)/2)),2)
            x_size_batch = x_size_tr[start_ind:start_ind + self.batch_size]
            c_batch = c_label_tr[start_ind:start_ind + self.batch_size]
            inds_batch = inds_db[start_ind:start_ind + self.batch_size]
            
            # if not self.set_loss:
            #     c_batch = np.vstack(c_batch)
            #     c1_batch = np.vstack(c1_batch)

            # Indexing and negative choosing
            # each item negative sampling
            selected_elements = []
            for row_ind in range(c_batch.shape[0]):
                tmp_elements = []
                for col_ind in range(c_batch.shape[1]):
                    label = c_batch[row_ind][col_ind]
                    if label != 11:
                        selected_ind = np.where(self.classtoind_dict[label] == inds_batch[row_ind][col_ind])
                        all_labeled_indices = self.class_dict[label][selected_ind][0]
                        top_75_percent_index = int(len(all_labeled_indices) * self.negative_level)
                        # top_75_percent_elements = all_labeled_indices[:top_75_percent_index]
                        top_75_percent_elements = all_labeled_indices[::-1][:top_75_percent_index]
                        sample_size = min(self.negative_item_num, len(top_75_percent_elements))
                        # Select random elements
                        negative_sample = np.random.choice(top_75_percent_elements, size=sample_size, replace=False)
                        # ----------------------------------------
                        # negative_sample = self.class_dict[label][selected_ind][0][: self.negative_item_num]
                        if len(negative_sample) < self.negative_item_num :
                            initial_value = self.class_dict[label][selected_ind][0][: self.negative_item_num][0]
        
                            # Calculate the number of elements to add
                            padding_length = self.negative_item_num - len(negative_sample)
                            
                            negative_sample = np.append(negative_sample, [initial_value] * padding_length)
                    else:# galleryで使われないラベル
                        negative_sample = self.class_dict[0][0][ : self.negative_item_num]
                        if len(negative_sample) < self.negative_item_num :
                            initial_value = negative_sample[0]
        
                            # Calculate the number of elements to add
                            padding_length = self.negative_item_num - len(negative_sample)
                            
                            negative_sample = np.append(negative_sample, [initial_value] * padding_length)
                    tmp_elements.append(np.reshape(self.x_train, [self.x_train.shape[0]*self.x_train.shape[1], self.dim])[negative_sample])
                selected_elements.append(tmp_elements)
            # 結果をテンソルに変換（形状: (41, self.negative_item_num, 4096)）
            negative_gallery = np.array(selected_elements)

            if not self.set_loss:
                yield (x_batch, x_size_batch, c_batch, negative_gallery), y_batch
            else:
                yield (x_batch, x_size_batch), y_batch
    def validation_generator(self):

        for index in range(0, len(self.inds_vr)-2, self.batch_size):
            # selected_elements = []
            # for label in range(self.max_label+1):  # ラベル0～self.negative_item_numに対して処理
            #     # 該当ラベルの要素のインデックスを取得
            #     label_indices = np.where(self.y_prevalid == label)[0]
            #     if label_indices.size != 0:
            #         # シャッフルしてランダムに抽出
            #         np.random.shuffle(label_indices)
            #         selected = label_indices[:self.negative_item_num]
                    
            #         # 足りない場合は-1で埋める
            #         if len(selected) < self.negative_item_num:
            #             selected = np.pad(selected, (0, self.negative_item_num - len(selected)), constant_values=-1)
                    
            #         # 選択した要素をリストに追加
            #         selected_elements.append(np.reshape(self.x_valid, [self.val_set_size*self.val_item_size, self.dim])[selected])
            #     else: # valid にはないカテゴリの処理
            #         selected_elements.append(np.zeros((self.negative_item_num,self.dim)))

            # # 結果をテンソルに変換（形状: (41, self.negative_item_num, 4096)）
            # negative_gallery = np.stack(selected_elements)
            start_ind = index

            batch_inds = self.inds_vr[start_ind:start_ind + self.batch_size]
            x_batch = self.x_valid[start_ind:start_ind + self.batch_size]
            y_batch = np.repeat(np.arange(int(len(x_batch)/2)),2)
            x_size_batch = self.x_size_valid[start_ind:start_ind + self.batch_size]
            c_batch = self.c_label_val[start_ind:start_ind + self.batch_size]
            inds_batch = self.inds_db_val[start_ind:start_ind + self.batch_size]

            selected_elements = []
            for row_ind in range(c_batch.shape[0]):
                tmp_elements = []
                for col_ind in range(c_batch.shape[1]):
                    label = c_batch[row_ind][col_ind]
                    if label != 11:
                        selected_ind = np.where(self.classtoind_dict_val[label] == inds_batch[row_ind][col_ind])
                        all_labeled_indices = self.class_dict_val[label][selected_ind][0]
                        top_75_percent_index = int(len(all_labeled_indices) * self.negative_level_val)
                        # top_75_percent_elements = all_labeled_indices[:top_75_percent_index]
                        top_75_percent_elements = all_labeled_indices[::-1][:top_75_percent_index]
                        sample_size = min(self.negative_item_num, len(top_75_percent_elements))
                        # Select random elements
                        negative_sample = np.random.choice(top_75_percent_elements, size=sample_size, replace=False)
                        # ----------------------------------------
                        # negative_sample = self.class_dict[label][selected_ind][0][: self.negative_item_num]
                        if len(negative_sample) < self.negative_item_num :
                            initial_value = self.class_dict_val[label][selected_ind][0][: self.negative_item_num][0]
        
                            # Calculate the number of elements to add
                            padding_length = self.negative_item_num - len(negative_sample)
                            
                            negative_sample = np.append(negative_sample, [initial_value] * padding_length)
                    else:# galleryで使われないラベル
                        negative_sample = self.class_dict_val[0][0][ : self.negative_item_num]
                        if len(negative_sample) < self.negative_item_num :
                            initial_value = negative_sample[0]
        
                            # Calculate the number of elements to add
                            padding_length = self.negative_item_num - len(negative_sample)
                            
                            negative_sample = np.append(negative_sample, [initial_value] * padding_length)
                    tmp_elements.append(np.reshape(self.x_valid, [self.x_valid.shape[0]*self.x_valid.shape[1], self.dim])[negative_sample])
                selected_elements.append(tmp_elements)
            # 結果をテンソルに変換（形状: (41, self.negative_item_num, 4096)）
            negative_gallery = np.array(selected_elements)
            if not self.set_loss:
                yield (x_batch, x_size_batch, c_batch, negative_gallery), y_batch
            else:
                yield (x_batch, x_size_batch), y_batch

            # yield (x_batch, x_size_batch), y_batch
    def test_generator(self):

        for index in range(0, len(self.inds_test)-34, self.batch_size):
            start_ind = index
            batch_inds = self.inds_test[start_ind:start_ind + self.batch_size]
            x_batch = self.x_valid[start_ind:start_ind + self.batch_size]
            y_batch = self.y_test[start_ind:start_ind + self.batch_size]
            # y_batch = y_batch.astype(np.float64)
            x_size_batch = self.x_size_valid[start_ind:start_ind + self.batch_size]
            c_batch = self.c_label_val[start_ind:start_ind + self.batch_size]
            inds_batch = self.inds_test[start_ind:start_ind + self.batch_size]
            id_batch = self.x_id_test[start_ind:start_ind + self.batch_size]
            if not self.set_loss:
                yield (x_batch, x_size_batch, c_batch, id_batch), y_batch
            else:
                yield (x_batch, x_size_batch), y_batch

    def get_train_dataset(self):
        if self.set_loss:
            return tf.data.Dataset.from_generator(
                self.train_generator,
                    output_types=((tf.float64, tf.float32), tf.float64),
                    output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,])), tf.TensorShape([None,]))
                    
            )
        else:
            return tf.data.Dataset.from_generator(
                self.train_generator,
                    output_types=((tf.float64, tf.float32, tf.int64, tf.float64), tf.float64),
                    output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,]), tf.TensorShape([None, self.max_item_num]), tf.TensorShape([None, self.max_item_num, self.negative_item_num, self.dim])), tf.TensorShape([None,]))
                    
            )
        
    
    def get_validation_dataset(self):     
        if self.set_loss:
            return tf.data.Dataset.from_generator(
                self.validation_generator,
                output_types=((tf.float64, tf.float32), tf.float64),
                output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,])), tf.TensorShape([None,]))
                    
            )
        else:
            return tf.data.Dataset.from_generator(
                self.validation_generator,
                output_types=((tf.float64, tf.float32, tf.int64, tf.float64), tf.float64),
                output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,]), tf.TensorShape([None, self.max_item_num]), tf.TensorShape([None, self.max_item_num, self.negative_item_num, self.dim])), tf.TensorShape([None,]))
                    
            )
    def get_test_dataset(self):     
        if self.set_loss:
            return tf.data.Dataset.from_generator(
                self.test_generator,
                output_types=((tf.float64, tf.float32), tf.float64),
                output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,])), tf.TensorShape([None,]))
                    
            )
        else:
            return tf.data.Dataset.from_generator(
                self.test_generator,
                output_types=((tf.float64, tf.float32, tf.int64, tf.int64), tf.float64),
                output_shapes=((tf.TensorShape([None, self.max_item_num, self.dim]), tf.TensorShape([None,]), tf.TensorShape([None, self.max_item_num]), tf.TensorShape([None, self.max_item_num])), tf.TensorShape([None,]))
                    
            )
'''
# generate lineared vector and whitening step
year = 2017
max_item_num = 5
batch_size = 100
pdb.set_trace()
test_generator = trainDataGenerator(year = year, batch_size = batch_size, max_item_num = max_item_num)
x_test, x_size_test, y_test, category1_test, category2_test, item_label_test, c2_test = test_generator.data_generation_train()
'''
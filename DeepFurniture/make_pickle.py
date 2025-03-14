import os
import pickle
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json
import pdb

def pad_features(features, target_num=8):
    current_num = features.shape[0]
    if current_num < target_num:
        padding = np.zeros((target_num - current_num, features.shape[1]), dtype=features.dtype)
        padded_features = np.vstack([features, padding])
    else:
        padded_features = features[:target_num]
        current_num = target_num
    return padded_features, current_num

def pad_categories(categories, target_num=8):
    current_num = categories.shape[0]
    if current_num < target_num:
        padding = np.zeros(target_num - current_num, dtype=int)  # デフォルトカテゴリID=0
        padded_categories = np.concatenate([categories, padding])
    else:
        padded_categories = categories[:target_num]
        current_num = target_num
    return padded_categories, current_num

def create_query_positive_groups(scene_features, min_items=6, max_items=16, max_item_num=8):
    X_Q = []
    X_P = []
    Y = []
    Y_cat_Q = []
    Y_cat_P = []
    X_sizes = []
    Y_sizes = []
    X_Item_id = []
    Y_Item_id = []
    # pdb.set_trace() ind = 0
    for scene_id, data in tqdm(scene_features.items(), desc='Processing scenes'):
        
        features = data['features']      # shape: (n_items, feature_dim)
        category_ids = data['category_ids']  # shape: (n_items,)
        item_ids = data['item_ids']
        
        num_items = features.shape[0]
        if num_items < min_items or num_items > max_items:
            continue
        queries_num = num_items // 2
        indices = list(range(num_items))
        random.shuffle(indices)
        query_indices = indices[:queries_num]
        positive_indices = indices[queries_num:]

        queries = features[query_indices]
        queries_cat = category_ids[query_indices]
        queries_item_id = item_ids[query_indices]
        positives = features[positive_indices]
        positives_cat = category_ids[positive_indices]
        positive_item_is = item_ids[positive_indices]

        queries_padded, q_size = pad_features(queries, target_num=max_item_num)
        queries_cat_padded, _ = pad_categories(queries_cat, target_num=max_item_num)
        positives_padded, p_size = pad_features(positives, target_num=max_item_num)
        positives_cat_padded, _ = pad_categories(positives_cat, target_num=max_item_num)
        queries_item_id_padded, _ = pad_categories(queries_item_id, target_num=max_item_num)
        positive_item_id_padded, _ = pad_categories(positive_item_is, target_num=max_item_num)

        X_Q.append(queries_padded)
        Y_cat_Q.append(queries_cat_padded)
        X_P.append(positives_padded)
        Y_cat_P.append(positives_cat_padded)
        Y.append(scene_id)
        X_sizes.append(q_size)  # クエリ側のアイテム数(元の数)を記録
        Y_sizes.append(p_size)
        X_Item_id.append(queries_item_id_padded)
        Y_Item_id.append(positive_item_id_padded)


    return np.array(X_Q), np.array(X_P), np.array(Y), np.array(Y_cat_Q), np.array(Y_cat_P), np.array(X_sizes), np.array(Y_sizes), np.array(X_Item_id), np.array(Y_Item_id)

def main():
    parser = argparse.ArgumentParser(description="Generate datasets for set retrieval.")
    parser.add_argument('--raw_data_path', type=str, required=True, help='Path to raw data pickle file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed datasets')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to be used as test set')
    parser.add_argument('--validation_size', type=float, default=0.1, help='Proportion of data to be used as validation set')
    parser.add_argument('--min_items', type=int, default=6, help='Minimum number of items in a scene')
    parser.add_argument('--max_items', type=int, default=16, help='Maximum number of items in a scene')
    parser.add_argument('--max_item_num', type=int, default=8, help='Maximum number of items to pad/truncate to')
    args = parser.parse_args()

    # データのロード
    with open(args.raw_data_path, 'rb') as f:
        scene_features = pickle.load(f)

    # データセットの生成
    X_Q, X_P, Y, Y_cat_Q, Y_cat_P, X_sizes, Y_sizes, X_Item_id, Y_Item_id = create_query_positive_groups(
        scene_features,
        min_items=args.min_items,
        max_items=args.max_items,
        max_item_num=args.max_item_num
    )
    
    # シーンIDを整数に変換
    unique_scene_ids = np.unique(Y)
    scene_id_to_int = {sid: idx for idx, sid in enumerate(unique_scene_ids)}
    Y_int = np.array([scene_id_to_int[sid] for sid in Y], dtype=np.int32)
    # Y_int = Y

    # カテゴリIDのエンコーディング
    # 既にカテゴリIDが整数である場合、必要に応じてインデックスを再マッピング
    unique_category_ids = np.unique(Y_cat_Q)  # クエリとポジティブで共通のカテゴリIDセット
    category_id_to_int = {cid: idx for idx, cid in enumerate(unique_category_ids)}
    Y_cat_Q_encoded = np.array([[category_id_to_int[cid] for cid in sample] for sample in Y_cat_Q], dtype=int)
    Y_cat_P_encoded = np.array([[category_id_to_int[cid] for cid in sample] for sample in Y_cat_P], dtype=int)

    # データを訓練用とテスト用に分割
    X_train_Q, X_temp_Q, X_train_P, X_temp_P, y_train, y_temp, y_train_cat_Q, y_temp_cat_Q, y_train_cat_P, y_temp_cat_P, x_sizes_train, x_sizes_temp, y_sizes_train, y_sizes_temp, x_item_id_train, x_item_id_tmp, y_item_id_train, y_item_id_tmp = train_test_split(
        X_Q, X_P, Y_int, Y_cat_Q_encoded, Y_cat_P_encoded, X_sizes, Y_sizes, X_Item_id, Y_Item_id, test_size=args.test_size + args.validation_size, random_state=42
    )
    val_size_relative = args.validation_size / (args.test_size + args.validation_size)
    X_val_Q, X_test_Q, X_val_P, X_test_P, y_val, y_test, y_val_cat_Q, y_test_cat_Q, y_val_cat_P, y_test_cat_P, x_sizes_val, x_sizes_test, y_sizes_val, y_sizes_test, x_item_id_val, x_item_id_test, y_item_id_val, y_item_id_test = train_test_split(
        X_temp_Q, X_temp_P, y_temp, y_temp_cat_Q, y_temp_cat_P, x_sizes_temp, y_sizes_temp, x_item_id_tmp, y_item_id_tmp, test_size=val_size_relative, random_state=42
    )
    
    # 出力ディレクトリの作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # データの保存
    with open(os.path.join(args.output_dir, 'train_fur.pkl'), 'wb') as f:
        pickle.dump((X_train_Q, X_train_P, y_train, y_train_cat_Q, y_train_cat_P, x_sizes_train, y_sizes_train), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.output_dir, 'validation_fur.pkl'), 'wb') as f:
        pickle.dump((X_val_Q, X_val_P, y_val, y_val_cat_Q, y_val_cat_P, x_sizes_val, y_sizes_val), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.output_dir, 'test_fur.pkl'), 'wb') as f:
        pickle.dump((X_test_Q, X_test_P, y_test, y_test_cat_Q, y_test_cat_P, x_sizes_test, y_sizes_test, x_item_id_test, y_item_id_test, scene_id_to_int), f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Dataset creation completed.")

if __name__ == '__main__':
    main()
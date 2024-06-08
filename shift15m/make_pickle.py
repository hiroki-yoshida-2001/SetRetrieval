import glob
import gzip
import json
import pickle
import pathlib
import tqdm
import pdb
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import sys, os


def get_trainvaltest_data(label_dir: str) -> Tuple[List, List, List]: #jsonを読むだけのコード
    path = pathlib.Path(label_dir)
    train = json.load(open(path / "train.json"))
    valid = json.load(open(path / "valid.json"))
    test = json.load(open(path / "test.json"))
    return train, valid, test


def get_labels(
    year: Union[str, int], split: int, data_root: str,
) -> Tuple[List, List, List]:
    
    # train.json, valid.json, test.jsonがあるディレクトリ
    label_dir = "!data/!"
    train, valid, test = get_trainvaltest_data(label_dir)
    

    return train, valid, test


def load_feature(path: str):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        feature = json.loads(f.read())
    return feature


# gzファイルを展開して読み込んでitem_featureをpklへ変換 + category_idごとのfeatureを作る
def save_pickles(
    year: Union[str, int], split: int, data_root: str, mode: str, label: List,
):
    # SHIFT15Mの outfit features.gzがあるディレクトリ
    feature_dir = "!dataset/features!"
    folder_name = f"{year}-{year}-split{split}/{mode}"
    output_dir = pathlib.Path(data_root) / "journal" / "pickles" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("saving pickle file to " + str(output_dir))
    category_id1_dict = {}
    category_id2_dict = {}

    for i in tqdm.tqdm(range(len(label))):
        set_data = label[i]

        set_id = set_data["set_id"]
        items = set_data["items"]
        features = []
        #----------------
        
        item_labels = []
        category_id1s = []
        category_id2s = []

        #----------------
        #pdb.set_trace()
        for item in items:
            category_id1 = item['category_id1']
            category_id2 = item['category_id2']
            item_id = item['item_id']
            feat_name = str(item["item_id"]) + ".json.gz"
            path = f"{feature_dir}/{feat_name}"

            # ---------------------------
            features.append(load_feature(path))
            item_labels.append(item_id)
            category_id1s.append(category_id1)
            category_id2s.append(category_id2)
            # ----------------------------
            
            item_feature = load_feature(path)

            # category_id1
            if category_id1 in category_id1_dict:
                category_id1_dict[category_id1].append({'item_id': item_id, 'set_id': set_id, 'feature': item_feature})
            else:
                category_id1_dict[category_id1] = [{'item_id': item_id, 'set_id': set_id, 'feature': item_feature}]
            # category_id2
            if category_id2 in category_id2_dict:
                category_id2_dict[category_id2].append({'item_id': item_id, 'set_id': set_id, 'feature': item_feature})
            else:
                category_id2_dict[category_id2] = [{'item_id': item_id, 'set_id': set_id, 'feature': item_feature}]
        # ----------------------------
        with open(output_dir / f"{set_id}.pkl", "wb") as f:
            pickle.dump(features, f)
            pickle.dump(category_id1s, f)
            pickle.dump(category_id2s, f)
            pickle.dump(item_labels, f)
        # ---------------------------
    '''
    category_id1_dict, category_id2_dict にはcategory_id1,2でまとめたset_id, item_idが辞書として格納
    unique_category_id1,2で category_id1の種類, category_id2の種類を調べる
    sorted_unique_category_id1,2でunique_category_id1,2をソートしてリスト化
    各カテゴリアイテムに対しては、category_id1_dict[sorted_unique_category_id1[i]] や category_id2_dict[sorted_unique_category_id2[i]] でアクセス可能※for i in range(len(sorted_unique_category)
    e.g, category_id1_dict[sorted_unique_category_id1[0]] => {'item_id': xxx, 'set_id' : yyy, 'feature': [zzz]※4096次元のベクトル}
    '''
    unique_category_id1 = set(category_id1_dict.keys())
    unique_category_id2 = set(category_id2_dict.keys())
    sorted_unique_category_id1 = sorted(list(unique_category_id1), reverse=False)
    sorted_unique_category_id2 = sorted(list(unique_category_id2), reverse=False)

    output_category_id1_dir = pathlib.Path(output_dir) / "category_id1"
    output_category_id1_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(sorted_unique_category_id1)):
        with open(output_category_id1_dir / f"{sorted_unique_category_id1[i]}.pkl", "wb") as f:
            pickle.dump(category_id1_dict[sorted_unique_category_id1[i]], f)

    output_category_id2_dir = pathlib.Path(output_dir) / "category_id2"
    output_category_id2_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(sorted_unique_category_id2)):
        with open(output_category_id2_dir / f"{sorted_unique_category_id2[i]}.pkl", "wb") as f:
            pickle.dump(category_id2_dict[sorted_unique_category_id2[i]], f)

    return

#-------------------------------
# gather and save pickle data
# train.pkl, valid.pkl, test.pklを作成する
def make_packed_pickle(files, save_path):
    X = []
    Y = []
    for file in files:
        with open(file, 'rb') as f:
            try:
                x = np.array(pickle.load(f))
            except:
                pass
                #print(f"{os.path.basename(file)}: empty")
            else:
                #print(f"{os.path.basename(file)}: {x.shape}")
                X.append(x)
                Y.append(int(os.path.basename(file).split('.')[0]))

    print(f"save to {save_path}")
    with open(save_path,'wb') as f:
        pickle.dump(X,f)
        pickle.dump(Y,f)
#-------------------------------



def main(args):
    # dataset
    train, valid, test = get_labels(args.year, args.split, args.data_root)

    save_pickles(args.year, args.split, args.data_root, "train", train)
    save_pickles(args.year, args.split, args.data_root, "valid", valid)
    save_pickles(args.year, args.split, args.data_root, "test", test)

    folder_name = f"{args.year}-{args.year}-split{args.split}"
    output_dir = pathlib.Path(args.data_root) / "journal" / "pickles" / folder_name
    train_path = f"{output_dir}/train"
    test_path = f"{output_dir}/test"
    valid_path = f"{output_dir}/valid"

    train_files = glob.glob(f"{train_path}/*.pkl")
    test_files = glob.glob(f"{test_path}/*.pkl")
    valid_files = glob.glob(f"{valid_path}/*.pkl")
    
    # save packed pickle
    make_packed_pickle(train_files, f"{output_dir}/train.pkl")
    make_packed_pickle(test_files, f"{output_dir}/test.pkl")
    make_packed_pickle(valid_files, f"{output_dir}/valid.pkl")

    item_seed_output = f"{output_dir}/item_seed"
    
    category2_pickle_path = glob.glob(f"{train_path}/category_id2/*.pkl")

    if not os.path.exists(category2_pickle_path):
        print("error: item feature pickle file divided by category_id2 does not exist")
        sys.exit()
    if not os.path.exists(item_seed_output):
        os.makedirs(item_seed_output)

    seed_vec = []
    for id in range(len(category2_pickle_path)):
        with open(category2_pickle_path[id], 'rb') as fp:
            category_id2_dict = pickle.load(fp)
        features = []
        for i in range(len(category_id2_dict)):
            features.append(category_id2_dict[i]["feature"])
        features = np.stack(features)
        seed_vec.append(np.mean(features, axis=0).tolist())

    seed_vec = np.stack(seed_vec)
    with open(f"{item_seed_output}/item_seed.pkl", "wb") as f:
        pickle.dump(seed_vec, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", "-y", type=int, default=2017)
    parser.add_argument("--split", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--data_root", type=str, default="data")

    args = parser.parse_args()

    main(args)

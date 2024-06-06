import glob
import gzip
import json
import pickle
import pathlib
import tqdm
import pdb
from typing import Any, Dict, List, Optional, Tuple, Union


def get_trainvaltest_data(label_dir: str) -> Tuple[List, List, List]: #jsonを読むだけのコード
    path = pathlib.Path(label_dir)
    train = json.load(open(path / "train.json"))
    valid = json.load(open(path / "valid.json"))
    test = json.load(open(path / "test.json"))
    return train, valid, test


def get_labels(
    year: Union[str, int], split: int, data_root: str,
) -> Tuple[List, List, List]:
   
    # train.json, valid.json, test.jsonがあるファイル 
    label_dir = "/data2/yoshida/mastermatching/data/"
    train, valid, test = get_trainvaltest_data(label_dir)
    

    return train, valid, test


def load_feature(path: str):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        feature = json.loads(f.read())
    return feature


def save_pickles(
    year: Union[str, int], split: int, data_root: str, mode: str, label: List,
):
    feature_dir = "! zozo-shift15m outfit features path !f"#/data2/nakamura/Datasets/zozo-shift15m/data/features"
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
        #pdb.set_trace()
        for item in items:
            category_id1 = item['category_id1']
            category_id2 = item['category_id2']
            item_id = item['item_id']
            feat_name = str(item["item_id"]) + ".json.gz"
            path = f"{feature_dir}/{feat_name}"
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
            # feat_name = str(item["item_id"]) + ".json.gz"
            # path = f"{feature_dir}/{feat_name}"
            # features.append(load_feature(path))
        # with open(output_dir / f"{id}.pkl", "wb") as f:
        #     pickle.dump(features, f)
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
    #pdb.set_trace()
    # category_id1_dict[sorted_unique_category_id1[0]][0] => exploit index 0 item id & set id whose category_id is sorted_unique_category_id1[0]
    # assert len(glob.glob(str(output_dir / "*"))) == len(label), "unmatched case"

    return


def main(args):
    # dataset
    train, valid, test = get_labels(args.year, args.split, args.data_root)

    save_pickles(args.year, args.split, args.data_root, "train", train)
    save_pickles(args.year, args.split, args.data_root, "valid", valid)
    save_pickles(args.year, args.split, args.data_root, "test", test)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", "-y", type=int, default=2017)
    parser.add_argument("--split", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--data_root", type=str, default="data")

    args = parser.parse_args()

    main(args)

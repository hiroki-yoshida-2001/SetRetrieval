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
    pdb.set_trace()
    # train.json, valid.json, test.jsonがあるファイル 
    label_dir = "/data2/yoshida/mastermatching/data/"
    train, valid, test = get_trainvaltest_data(label_dir)
    pdb.set_trace()

    return train, valid, test


def load_feature(path: str):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        feature = json.loads(f.read())
    return feature


def save_pickles(
    year: Union[str, int], split: int, data_root: str, mode: str, label: List,
):
    feature_dir = f"/data2/nakamura/Datasets/zozo-shift15m/data/features"
    folder_name = f"{year}-{year}-split{split}/{mode}"
    output_dir = pathlib.Path(data_root) / "forpack" / "pickles" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("saving pickle file to " + str(output_dir))
    #pdb.set_trace()
    for i in tqdm.tqdm(range(len(label))):
        set_data = label[i]

        id = set_data["set_id"]
        items = set_data["items"]
        features = []
        item_labels = []
        category_id1s = []
        category_id2s = []
        for item in items:
            category_id1 = item['category_id1']
            category_id2 = item['category_id2']
            item_id = item['item_id']
            feat_name = str(item["item_id"]) + ".json.gz"
            path = f"{feature_dir}/{feat_name}"
            features.append(load_feature(path))
            item_labels.append(item_id)
            category_id1s.append(category_id1)
            category_id2s.append(category_id2)
        with open(output_dir / f"{id}.pkl", "wb") as f:
            pickle.dump(features, f)
            pickle.dump(category_id1s, f)
            pickle.dump(category_id2s, f)
            pickle.dump(item_labels, f)

    assert len(glob.glob(str(output_dir / "*"))) == len(label), "unmatched case"
    
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

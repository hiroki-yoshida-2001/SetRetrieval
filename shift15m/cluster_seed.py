import glob
import gzip
import json
import pickle
import pathlib
import tqdm
import pdb
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import glob
import os
from pathlib import Path
from PIL import Image


# category_idごとにまとめたpickleファイルを読み込んで、可視化などを行うプログラム
category1_pickle_path = glob.glob("/data2/yoshida/mastermatching/data/journal/pickles/2017-2017-split0/train/category_id1/*.pkl")
category2_pickle_path = glob.glob("/data2/yoshida/mastermatching/data/journal/pickles/2017-2017-split0/train/category_id2/*.pkl")
IQONPath = '/data2/yoshida/mastermatching/data/IQON/IQON3000'

Visulaize_output_category1 = "/data2/yoshida/mastermatching/set_ret/set_rep_vec_asym_attention/shift15m/visualize/category1"
Visulaize_output_category2 = "/data2/yoshida/mastermatching/set_ret/set_rep_vec_asym_attention/shift15m/visualize/category2"

Visualize = False

if not os.path.exists(Visulaize_output_category1):
    os.makedirs(Visulaize_output_category1)
if not os.path.exists(Visulaize_output_category2):
    os.makedirs(Visulaize_output_category2)

if Visualize:
    # load pickle data ("set_id", "item_id", features) for each category1 
    for id in range(len(category1_pickle_path)):
        with open(category1_pickle_path[id], 'rb') as fp:
            category_id1_dict = pickle.load(fp)
        # ----------------------------------------------------
        os.chdir(IQONPath)
        print(f"searching items matching category_id1 from {IQONPath}")
        images = []
        all_item_label = 0
        for item_ind in range(len(category_id1_dict)):
            set_id = category_id1_dict[item_ind]["set_id"]
            item_id = category_id1_dict[item_ind]["item_id"]

            # 目的のディレクトリ名を指定
            target_directory_name = str(set_id)
            target_item_name = str(item_id)

            # 現在のディレクトリから再帰的に目的のディレクトリを探す
            if len(images) < 21:
                for path in Path(".").rglob(target_directory_name):
                    if path.is_dir():
                        print("Found directory:", path)
                        estimate_file_paths = os.path.join(f"{IQONPath}/{path}/", f"{target_item_name}")
                        jpg_file_paths = glob.glob(f"{estimate_file_paths}*.jpg")
                        if len(jpg_file_paths) > 0:
                            img = Image.open(jpg_file_paths[0])
                            images.append(img)
            else:
                concatenated_image = Image.new("RGB", (sum(img.width for img in images), max(img.height for img in images)))
                x_offset = 0
                for img in images:
                    concatenated_image.paste(img, (x_offset, 0))
                    x_offset += img.width

                # 連結した画像を保存
                print(f"savaing item image to {Visulaize_output_category1}/category_id1_{id}.jpg")
                concatenated_image.save(Visulaize_output_category1+"/category_id1_{}.jpg".format(id))
                break

            all_item_label += 1

        # ----------------------------------------------------

    for id in range(len(category2_pickle_path)):
        with open(category2_pickle_path[id], 'rb') as fp:
            category_id2_dict = pickle.load(fp)
        # ----------------------------------------------------
        os.chdir(IQONPath)
        print(f"searching items matching category_id2 from {IQONPath}")
        images = []
        all_item_label = 0
        for item_ind in range(len(category_id2_dict)):
            set_id = category_id2_dict[item_ind]["set_id"]
            item_id = category_id2_dict[item_ind]["item_id"]

            # 目的のディレクトリ名を指定
            target_directory_name = str(set_id)
            target_item_name = str(item_id)

            # 現在のディレクトリから再帰的に目的のディレクトリを探す
            if len(images) < 21:
                for path in Path(".").rglob(target_directory_name):
                    if path.is_dir():
                        print("Found directory:", path)
                        estimate_file_paths = os.path.join(f"{IQONPath}/{path}/", f"{target_item_name}")
                        jpg_file_paths = glob.glob(f"{estimate_file_paths}*.jpg")
                        
                        if len(jpg_file_paths) > 0:
                            img = Image.open(jpg_file_paths[0])
                            images.append(img)
            else:
                concatenated_image = Image.new("RGB", (sum(img.width for img in images), max(img.height for img in images)))
                x_offset = 0
                for img in images:
                    concatenated_image.paste(img, (x_offset, 0))
                    x_offset += img.width

                # 連結した画像を保存
                print(f"savaing item image to {Visulaize_output_category2}/category_id2_{id}.jpg")
                concatenated_image.save(Visulaize_output_category2+"/category_id2_{}.jpg".format(id))
                break

            all_item_label += 1

# ----------------------------------------------------
# pdb.set_trace()

# features = []
# for i in range(len(category_id1_dict)):
#     features.append(category_id1_dict[i]["feature"])
# pdb.set_trace()
# features = np.stack(features)

item_seed_output = "/data2/yoshida/mastermatching/set_ret/set_rep_vec_asym_attention/shift15m/pickle_data/item_seed"
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


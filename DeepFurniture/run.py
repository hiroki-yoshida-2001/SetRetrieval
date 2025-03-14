import tensorflow as tf
import matplotlib.pylab as plt
import os
import numpy as np
import pdb
import copy
import pickle
import sys
import argparse
import make_dataset as data
sys.path.insert(0, "../")
import models_fur
import util
from pathlib import Path
import glob
from PIL import Image, ImageDraw, ImageFont

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

import SMscore_model


#----------------------------
# set parameters

# get options
parser = util.parser_run()
args = parser.parse_args()


# 4096次元の候補ベクトルの次元削減FC層を学習するか否か
is_Cvec_linear = args.is_Cvec_linear
# year of data and max number of items
year = 2017
max_item_num = 5
test_cand_num = 5

# number of epochs
epochs = 100

# early stoppoing parameter
patience = 5

# batch size
batch_size = args.batch_size

# number of representive vectors 5=>41
rep_vec_num = 11


# set random seed (gpu)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.random.seed(args.trial)
tf.random.set_seed(args.trial)
#----------------------------


# ---------------------------


#----------------------------
# make Path

# make experiment path containing CNN and set-to-set model
# experiment path を動的に作成する
experimentPath = 'Experiment_DeepFurniture'

if args.pretrained_mlp:
    experimentPath = f"{experimentPath}_TrainMLP"

# loss method (gallerytype)
gallerytype = util.gallery_type(args.gallerytype)

experimentPath = os.path.join(experimentPath, gallerytype)

# loss method (style)
style_method = util.style_method(args.style_loss)
experimentPath = os.path.join(experimentPath, style_method)

# make set-to-set model path
modelPath = experimentPath

if args.is_set_norm:
    modelPath += f'_setnorm'

modelPath = os.path.join(modelPath, f"negativesch{args.negative_scheduling}")
modelPath = os.path.join(modelPath, f"is_l2_loss{args.is_l2_loss}")
modelPath = os.path.join(modelPath, f"batch_size{batch_size}")
modelPath = os.path.join(modelPath,f"year{year}")
modelPath = os.path.join(modelPath,f"max_item_num{max_item_num}")
modelPath = os.path.join(modelPath,f"layer{args.num_layers}")
modelPath = os.path.join(modelPath,f"num_head{args.num_heads}")
modelPath = os.path.join(modelPath,f"is_Cvec_linear_{is_Cvec_linear}")
modelPath = os.path.join(modelPath, f"use_c1_label{args.label_ver}")
modelPath = os.path.join(modelPath, f"category_emb{args.category_emb}")

if not os.path.exists(modelPath):
    path = os.path.join(modelPath,'model')
    os.makedirs(path)

    path = os.path.join(modelPath,'result')
    os.makedirs(path)


#----------------------------
# load init seed vectors
init_seed_pickle_path = "furniture_seed.pkl"
whitening_path = "furniture_gausenoise.pkl"
train_generator = data.DataGenerator(year=year, batch_size=args.batch_size, max_item_num=max_item_num,set_loss=False, whitening_path=not os.path.exists(whitening_path), seed_path=not os.path.exists(init_seed_pickle_path))
train_data_set = train_generator.get_train_dataset()
validation_data_set = train_generator.get_validation_dataset()
test_data_set = train_generator.get_test_dataset()


if not os.path.exists(init_seed_pickle_path):
    print("init_seed vectors haven't been generated ")
    seed_vectors = 0
else:
    print("Loading init_seed vectors...")
    with open(init_seed_pickle_path,'rb') as fp:
        seed_vectors = pickle.load(fp)
    
    rep_vec_num  = len(seed_vectors)
    seed_vectors = seed_vectors.tolist()

# set-matching model net 
# ------------------------
# # SetMatchingModelの定義と学習
set_matching_model = SMscore_model.SetMatchingModel(
    isCNN=False,                     # CNNを使用するかどうか
    is_set_norm=args.is_set_norm,    # Set Normalizationの有無
    is_cross_norm=True,# Cross Normalizationの有無
    is_final_linear=True,            # 最終的な線形層を使用するかどうか（デフォルト: True）
    num_layers=args.num_layers,      # エンコーダ・デコーダのレイヤー数
    num_heads=args.num_heads,        # Attentionのヘッド数
    mode='setRepVec_pivot',          # モード（デフォルト: 'setRepVec_pivot'）
    baseChn=128,            # ベースのチャンネル数
    rep_vec_num=3,                   # 代表ベクトルの数
    seed_init = seed_vectors,                   # シードの初期化
    cnn_class_num=2,                 # CNNの分類クラス数（デフォルト: 2）
    max_channel_ratio=2,             # チャンネル倍率
    is_neg_down_sample=True, # ネガティブサンプルのダウンサンプリングを使用するかどうか
    Whitening_path=whitening_path,
    pretrain = True
)

# マッチングモデル学習
if args.train_matching:
    SetMatching_model_path = f"{experimentPath}_TrainScore"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(SetMatching_model_path, monitor='val_binary_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
    cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
    # model mlp
    # model_mlp.load_weights(mlp_checkpoint_path)
    # down load mlp param
    # set_matching_model.MLP = model_mlp
    set_matching_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'], run_eagerly=True)
    train_data_set = train_data_set.prefetch(1)
    validation_data_set = validation_data_set.prefetch(1)
    set_matching_model.fit(train_data_set, epochs=epochs, validation_data=validation_data_set, callbacks=[cp_callback, cp_earlystopping])

    # 事前学習モデルの保存
    set_matching_model.save_weights('SetmatchingModelscore/set_matching_weights.ckpt')
else:
    set_matching_model.load_weights('SetmatchingModelscore/set_matching_weights.ckpt')

# ------------------------
# set-matching network
model_smn = models_fur.SMN(is_set_norm=args.is_set_norm, num_layers=args.num_layers, num_heads=args.num_heads, baseChn=args.baseChn, rep_vec_num=rep_vec_num, seed_init = seed_vectors, is_category_emb=args.category_emb, set_loss=args.set_loss, c1_label=args.label_ver, gallerytype=gallerytype, style_loss=style_method, L2_norm_loss=args.is_l2_loss, whitening=whitening_path)

checkpoint_path = os.path.join(modelPath,"model/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_Set_accuracy', save_weights_only=True, mode='min', save_best_only=True, save_freq='epoch', verbose=1)
cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_Set_accuracy', patience=patience, mode='min', min_delta=0.001, verbose=1)
result_path = os.path.join(modelPath,"result/result.pkl")

if not os.path.exists(result_path):

    # print("load MLP model")
    # model_mlp.load_weights(mlp_checkpoint_path)
    # model_smn.MLP = model_mlp
    model_smn.SetMatchingModel = set_matching_model

    # setting training, loss, metric to model
    if gallerytype == 'InBatch': # zozo + alpha loss
        model_smn.compile(optimizer="adam", loss=util.InBatchCLIPLoss, metrics=util.Retrieval_acc, run_eagerly=True)
    else: # proposed loss 
        model_smn.compile(optimizer="adam", loss=util.OutBatchCLIPLoss, metrics=util.Retrieval_acc, run_eagerly=True)
    # execute training
    train_data_set = train_data_set.prefetch(1)
    validation_data_set = validation_data_set.prefetch(1)
    history = model_smn.fit(train_data_set, epochs=epochs, validation_data=validation_data_set, shuffle=True, callbacks=[cp_callback,cp_earlystopping])
    
    # accuracy and loss
    acc = history.history['Set_accuracy']
    val_acc = history.history['val_Set_accuracy']
    match_loss = history.history['Match_loss']
    val_match_loss = history.history['val_Match_loss']
    cos_sim_loss = history.history['cos_sim_loss']
    val_cos_sim_loss = history.history['val_cos_sim_loss']
    L2_loss = history.history['L2_loss']
    val_L2_loss = history.history['val_L2_loss']

    # plot loss & acc
    # util.plotLossACC(modelPath,loss,val_loss,acc,val_acc)
    
    # dump to pickle
    with open(result_path,'wb') as fp:
        pickle.dump(acc,fp)
        pickle.dump(val_acc,fp)
        pickle.dump(match_loss,fp)
        pickle.dump(val_match_loss,fp)
        pickle.dump(cos_sim_loss, fp)
        pickle.dump(val_cos_sim_loss, fp)
        pickle.dump(L2_loss, fp)
        pickle.dump(val_L2_loss, fp)
else:
    # load trained parameters
    print("load models")
    # model_mlp.load_weights(mlp_checkpoint_path)
    # model_smn.MLP = model_mlp
    model_smn.SetMatchingModel = set_matching_model
    model_smn.load_weights(checkpoint_path)
#----------------------------

#---------------------------------

test_data_path = os.path.join(modelPath, "result/predict_vector_test.pkl")
if not os.path.exists(test_data_path):  
    train_generator = data.DataGenerator(year=year, batch_size=98, max_item_num=max_item_num, use_all_pred=False) 
    test_data_set = train_generator.get_test_dataset()
    # set data generator for evaluation and test (similar settings using test.pkl as train and valid)
    test_data_set = test_data_set.prefetch(1)

    # model_mlp.load_weights(mlp_checkpoint_path)
    # model_smn.MLP = model_mlp
    model_smn.SetMatchingModel = set_matching_model
    model_smn.compile(optimizer='adam',loss=util.OutBatchCLIPLoss, metrics=util.Retrieval_acc,run_eagerly=True)

    # test_loss, test_acc = model_smn.evaluate((x_test,x_size_test, [], c2_test),y_test,batch_size=batch_size,verbose=1) 
    x_test, x_size_test, c_label_test, y_test, ans_c_label_test, test_pred_vec, gallery, test_item_id = model_smn.predict((test_data_set),batch_size=100, verbose=1)
    
    with open(test_data_path,'wb') as fp:
        pickle.dump(test_pred_vec, fp)
        pickle.dump(gallery, fp)
        pickle.dump(y_test, fp)
        pickle.dump(c_label_test, fp)
        pickle.dump(ans_c_label_test,fp)
        pickle.dump(test_item_id, fp)
else:
    with open(test_data_path, 'rb') as fp:
        test_pred_vec = pickle.load(fp)
        gallery = pickle.load(fp)
        y_test = pickle.load(fp)
        c_label_test = pickle.load(fp)
        ans_c_label_test = pickle.load(fp)
        test_item_id = pickle.load(fp)

test_rank = False
rank_path = os.path.join(modelPath, "result/ranking_Test.pkl")
if test_rank:
    rank = util.compute_test_rank(gallery, test_pred_vec, y_test, c_label_test)
    with open(rank_path, 'wb') as fp:
        pickle.dump(rank, fp)
# predict_rank = np.min(rank, axis=-1)

# Top-Kaccuracy metrics 
util.ranking_analysis(Ranking_pkl_path=rank_path, data_path=test_data_path, Dataset='DeepFurniture')


Test_match_score = False
path = os.path.join(modelPath, "result/Matchscore_test.pkl")
if Test_match_score:
    train_generator = data.DataGenerator(year=year, batch_size=28, max_item_num=max_item_num, use_all_pred=False)
    test_data_set = train_generator.get_test_dataset()
    test_pred_vec, gallery, Real_score, Fake_score = model_smn.predict((test_data_set),batch_size=50, verbose=1)

    # 真の組み合わせと予測の組み合わせを同時に保存
    # Test の時は　最近傍アイテムで考えるべき？
    with open(path, 'wb') as fp:
        pickle.dump(Real_score, fp)
        pickle.dump(Fake_score, fp)


# Visualize Furniture Cordinate Set
visualize = False
if visualize:

    image_dir = "/data1/yamazono/research/DeepFurniture/uncompressed_data/furnitures"
    save_dir = "/data1/yoshida/setMatching/shift15m/Deepfurniture/visualization_results"

    util.visualize_set_top3(test_data_path, image_dir, save_dir)
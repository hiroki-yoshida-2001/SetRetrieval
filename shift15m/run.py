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
import models
import util
from pathlib import Path
import glob
from PIL import Image, ImageDraw, ImageFont

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


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
rep_vec_num = 41


# set random seed (gpu)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.random.seed(args.trial)
tf.random.set_seed(args.trial)
#----------------------------


# ---------------------------
# load init seed vectors
pickle_path = "pickle_data"
init_seed_pickle_path = os.path.join(pickle_path, "item_seed/item_seed.pkl")

if not os.path.exists(init_seed_pickle_path):
    print("init_seed vectors haven't been generated ")
    seed_vectors = 0
else:
    print("Loading init_seed vectors...")
    with open(init_seed_pickle_path,'rb') as fp:
        seed_vectors = pickle.load(fp)
    
    rep_vec_num  = len(seed_vectors)
    seed_vectors = seed_vectors.tolist()


#----------------------------
# make Path

# make experiment path containing CNN and set-to-set model
# experiment path を動的に作成する
experimentPath = 'Experiment'
if args.pretrained_mlp:
    experimentPath = f"{experimentPath}_TrainMLP"

if not os.path.exists(experimentPath):
    os.makedirs(experimentPath)

# make set-to-set model path
modelPath = experimentPath

if args.is_set_norm:
    modelPath += f'_setnorm'

modelPath = os.path.join(modelPath, f"batch_size{batch_size}")
modelPath = os.path.join(modelPath,f"year{year}")
modelPath = os.path.join(modelPath,f"max_item_num{max_item_num}")
modelPath = os.path.join(modelPath,f"layer{args.num_layers}")
modelPath = os.path.join(modelPath,f"num_head{args.num_heads}")
modelPath = os.path.join(modelPath,f"{args.trial}")
modelPath = os.path.join(modelPath,f"is_Cvec_linear_{is_Cvec_linear}")
modelPath = os.path.join(modelPath, f"use_c1_label{args.label_ver}")
modelPath = os.path.join(modelPath, f"category_emb{args.category_emb}")

if not os.path.exists(modelPath):
    path = os.path.join(modelPath,'model')
    os.makedirs(path)

    path = os.path.join(modelPath,'result')
    os.makedirs(path)
#----------------------------

#----------------------------
# make data
if args.pretrained_mlp:
    train_generator = data.trainDataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num, mlp_flag=args.pretrained_mlp)
    x_valid, y_valid = train_generator.data_generation_val()
    x_train, y_train = train_generator.data_generation_train()

    # pdb.set_trace()
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)

    # クラス重みを計算
    class_weights = {i: total_samples / count for i, count in enumerate(class_counts) if count > 0}

    baseMLPChn = 512
    # Multi layer perceptron model (pretrained network)
    model_mlp = models.MLP(baseChn=baseMLPChn, category_class_num=len(seed_vectors))

    mlp_path = os.path.join(modelPath, f'Pretrained_MLP_{baseMLPChn}')
    #mlp_path = "/data2/yoshida/mastermatching/set_ret/set_rep_vec_asym_attention/shift15m/experiment_mlptmp_TrainMLP/setRepVec_pivot_64_setnorm_crossnorm/year2017/max_item_num5/layer3/num_head5/1/use_Cvec_1/is_Cvec_linear_0/calc_set_simBERTscore/Pretrained_MLP_512"
    mlp_checkpoint_path = os.path.join(mlp_path,"model/cp.ckpt")
    mlp_checkpoint_dir = os.path.dirname(mlp_checkpoint_path)
    mlp_cp_callback = tf.keras.callbacks.ModelCheckpoint(mlp_checkpoint_path, monitor='val_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
    mlp_cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
    mlp_result_path = os.path.join(modelPath,"result/result.pkl")
    pdb.set_trace()
    if not os.path.exists(mlp_result_path):

        model_mlp.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
        # execute training
        history = model_mlp.fit(train_generator, epochs=epochs, validation_data=(x_valid, y_valid), shuffle=True, class_weight=class_weights, callbacks=[mlp_cp_callback,mlp_cp_earlystopping])

        # accuracy and loss
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # plot loss & acc
        util.plotLossACC(mlp_path,loss,val_loss,acc,val_acc)

        # dump to pickle
        pdb.set_trace()
        with open(mlp_result_path,'wb') as fp:
            pickle.dump(acc,fp)
            pickle.dump(val_acc,fp)
            pickle.dump(loss,fp)
            pickle.dump(val_loss,fp)
    if args.tf_data:
        train_generator = data.DataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num)
        train_data_set = train_generator.get_train_dataset()
        validation_data_set = train_generator.get_validation_dataset()
    else:
        train_generator = data.trainDataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num)
        x_valid, x_size_valid, y_valid = train_generator.data_generation_val()
else:
    baseMLPChn = 512
    model_mlp = models.MLP(baseChn=baseMLPChn, category_class_num=len(seed_vectors))
    mlp_path = "~/Pretrained_MLP_512" # please Specify MLP path manually...
    mlp_checkpoint_path = os.path.join(mlp_path,"model/cp.ckpt")
    mlp_checkpoint_dir = os.path.dirname(mlp_checkpoint_path)
    mlp_cp_callback = tf.keras.callbacks.ModelCheckpoint(mlp_checkpoint_path, monitor='val_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
    mlp_cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
    mlp_result_path = os.path.join(modelPath,"result/result.pkl")
    
    if args.tf_data:
        train_generator = data.DataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num, use_all_pred=False)
        train_data_set = train_generator.get_train_dataset()
        validation_data_set = train_generator.get_validation_dataset()
    else:
        train_generator = data.trainDataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num, use_all_pred=False)
        #x_valid, x_size_valid, y_valid = train_generator.data_generation_val()
        x_valid, x_size_valid, y_valid, category1_valid, category2_valid, item_label_valid, c2_valid = train_generator.data_generation_val()



# set data generator for test
# test_generator = data.testDataGenerator(year = year, cand_num = test_cand_num)
# x_test = test_generator.x
# x_size_test = test_generator.x_size
# y_test = test_generator.y
# test_batch_size = test_generator.batch_grp_num
#----------------------------

# set-matching network

model_smn = models.SMN(isCNN=False, is_TrainableMLP=True, is_set_norm=args.is_set_norm, num_layers=args.num_layers, num_heads=args.num_heads, baseChn=args.baseChn, baseMlp = baseMLPChn, rep_vec_num=rep_vec_num, seed_init = seed_vectors, is_Cvec_linear=is_Cvec_linear, is_category_emb=args.category_emb, use_all_pred=args.use_all_pred, c1_label=args.label_ver)

checkpoint_path = os.path.join(modelPath,"model/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_Set_accuracy', save_weights_only=True, mode='min', save_best_only=True, save_freq='epoch', verbose=1)
cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_Set_accuracy', patience=patience, mode='min', min_delta=0.001, verbose=1)
result_path = os.path.join(modelPath,"result/result.pkl")

if not os.path.exists(result_path):

    print("load MLP model")
    model_mlp.load_weights(mlp_checkpoint_path)
    model_smn.MLP = model_mlp

    # setting training, loss, metric to model
    model_smn.compile(optimizer="adam", loss=util.Set_item_Cross_entropy, metrics=models.CustomMetric(), run_eagerly=True)
    # execute training
    if args.tf_data:
        train_data_set = train_data_set.prefetch(1)
        validation_data_set = validation_data_set.prefetch(1)
        pdb.set_trace()
        history = model_smn.fit(train_data_set, epochs=epochs, validation_data=validation_data_set, shuffle=True, callbacks=[cp_callback,cp_earlystopping])
    else:
        pdb.set_trace()
        if args.use_all_pred:
            history = model_smn.fit(train_generator, epochs=epochs, validation_data=((x_valid, x_size_valid), y_valid), shuffle=True, callbacks=[cp_callback,cp_earlystopping])
        else:
            history = model_smn.fit(train_generator, epochs=epochs, validation_data=((x_valid, x_size_valid, c2_valid), y_valid), shuffle=True, callbacks=[cp_callback,cp_earlystopping])
    # accuracy and loss
    acc = history.history['Set_accuracy']
    val_acc = history.history['val_Set_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot loss & acc
    util.plotLossACC(modelPath,loss,val_loss,acc,val_acc)

    # dump to pickle
    with open(result_path,'wb') as fp:
        pickle.dump(acc,fp)
        pickle.dump(val_acc,fp)
        pickle.dump(loss,fp)
        pickle.dump(val_loss,fp)
else:
    # load trained parameters
    print("load models")
    model_smn.load_weights(checkpoint_path)
#----------------------------
pdb.set_trace()
#---------------------------------

path = os.path.join(modelPath, "result/predict_vector.pkl")
if not os.path.exists(path):   
    # set data generator for evaluation and test (similar settings using test.pkl as train and valid)
    test_generator = data.trainDataGenerator(year = year, batch_size = batch_size, max_item_num = max_item_num)
    x_test, x_size_test, y_test, category1_test, category2_test, item_label_test, c2_test = test_generator.data_generation_test()

    model_smn.compile(optimizer='adam',loss=model_smn.item_hinge_loss, metrics=models.CustomMetric(),run_eagerly=True)

    # test_loss, test_acc = model_smn.evaluate((x_test,x_size_test, [], c2_test),y_test,batch_size=batch_size,verbose=1) 
    test_pred_vec, gallery, replicated_set_label, query_id = model_smn.predict((x_test,x_size_test, c2_test, y_test, item_label_test),batch_size=len(x_test), verbose=1)
    
    with open(path,'wb') as fp:
        pickle.dump(test_pred_vec, fp)
        pickle.dump(gallery, fp)
        pickle.dump(y_test, fp)
        pickle.dump(replicated_set_label, fp)
        pickle.dump(query_id, fp)
        pickle.dump(category1_test,fp)
        pickle.dump(item_label_test, fp)
else:
    with open(path, 'rb') as fp:
        test_pred_vec = pickle.load(fp)
        gallery = pickle.load(fp)
        y_test = pickle.load(fp)
        replicated_set_label = pickle.load(fp)
        query_id = pickle.load(fp)
        category1_test = pickle.load(fp)
        item_label_test = pickle.load(fp)

test_rank = True
path = os.path.join(modelPath, "result/ranking.pkl")
if test_rank:
    rank = util.compute_test_rank(gallery, test_pred_vec, y_test, category1_test)
    
    with open(path, 'wb') as fp:
        pickle.dump(rank, fp)
else:
    with open(path, 'rb') as fp:
        rank = pickle.load(fp)
   
rank = np.array(rank)
predict_rank = rank
# predict_rank = np.min(rank, axis=-1)

averages = []
for row in predict_rank:
    positive_elements = row[row != -1]
    average = np.mean(positive_elements)
    averages.append(average)

# 平均値のリストを配列に変換
averages_array = np.array(averages)
nan_pd_array = np.where(averages_array == -1, np.nan, averages_array)
# mean_values = np.nanmean(nan_pd_array, axis=1)
mean_values = np.nanmean(nan_pd_array, axis=-1)
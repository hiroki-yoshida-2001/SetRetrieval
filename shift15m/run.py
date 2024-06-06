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
from PIL import Image

#----------------------------
# set parameters

# get options
parser = util.parser_run()
args = parser.parse_args()

# mode name
mode = util.mode_name(args.mode)

# setscore_func choice (アイテム間類似度=>集合間の類似度 の関数)
calc_set_sim = util.calc_set_sim_name(args.calc_set_sim)

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
batch_size = 50

# number of representive vectors 5=>41
rep_vec_num = 41

# negative down sampling
is_neg_down_sample = True

# set random seed (gpu)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

    if not args.use_Cvec:
        seed_vectors = 0

#----------------------------
# make Path

# make experiment path containing CNN and set-to-set model
# experiment path を動的に作成する
pdb.set_trace()
experimentPath = 'experiment'
if not os.path.exists(experimentPath):
    os.makedirs(experimentPath)

# make set-to-set model path
modelPath = os.path.join(experimentPath, f'{mode}_{args.baseChn}')

if args.is_set_norm:
    modelPath += f'_setnorm'

if args.is_cross_norm:
    modelPath += f'_crossnorm' 

modelPath = os.path.join(modelPath,f"year{year}")
modelPath = os.path.join(modelPath,f"max_item_num{max_item_num}")
modelPath = os.path.join(modelPath,f"layer{args.num_layers}")
modelPath = os.path.join(modelPath,f"num_head{args.num_heads}")
modelPath = os.path.join(modelPath,f"{args.trial}")
modelPath = os.path.join(modelPath,f"use_Cvec_{args.use_Cvec}")
modelPath = os.path.join(modelPath,f"is_Cvec_linear_{is_Cvec_linear}")
modelPath = os.path.join(modelPath,f"calc_set_sim{calc_set_sim}")
if not os.path.exists(modelPath):
    path = os.path.join(modelPath,'model')
    os.makedirs(path)

    path = os.path.join(modelPath,'result')
    os.makedirs(path)
#----------------------------

#----------------------------
# make data
train_generator = data.trainDataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num)
x_valid, x_size_valid, y_valid = train_generator.data_generation_val()


# set data generator for test
# test_generator = data.testDataGenerator(year = year, cand_num = test_cand_num)
# x_test = test_generator.x
# x_size_test = test_generator.x_size
# y_test = test_generator.y
# test_batch_size = test_generator.batch_grp_num
#----------------------------

# set data generator for evaluation and test (similar settings using test.pkl as train and valid)
test_generator = data.trainDataGenerator(year = year, batch_size = batch_size, max_item_num = max_item_num)
x_test, x_size_test, y_test, category1_test, category2_test, item_label_test = test_generator.data_generation_test()
#----------------------------

# set-matching network
model_smn = models.SMN(isCNN=False, is_final_linear=True, is_set_norm=args.is_set_norm, is_cross_norm=args.is_cross_norm, num_layers=args.num_layers, num_heads=args.num_heads, baseChn=args.baseChn, mode=mode, calc_set_sim=calc_set_sim, rep_vec_num=rep_vec_num, seed_init = seed_vectors, is_neg_down_sample=is_neg_down_sample, is_Cvec_linear=is_Cvec_linear)

checkpoint_path = os.path.join(modelPath,"model/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_Set_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_Set_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
result_path = os.path.join(modelPath,"result/result.pkl")

if not os.path.exists(result_path):

    # setting training, loss, metric to model
    model_smn.compile(optimizer="adam", loss=util.Set_hinge_loss, metrics=util.Set_accuracy, run_eagerly=True)
    # execute training
    history = model_smn.fit(train_generator, epochs=epochs, validation_data=((x_valid, x_size_valid), y_valid), shuffle=True, callbacks=[cp_callback,cp_earlystopping])

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




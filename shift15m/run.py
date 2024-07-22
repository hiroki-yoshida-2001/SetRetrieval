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

test_batch_size = 7700

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
experimentPath = 'experiment_mlp'

if args.pretrained_mlp:
    experimentPath = f"{experimentPath}_TrainMLP"

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
if args.pretrained_mlp:
    train_generator = data.trainDataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num, mlp_flag=args.pretrained_mlp)
    x_valid, y_valid = train_generator.data_generation_val()
    x_train, y_train = train_generator.data_generation_train()

    class_counts = np.bincount(y_train)
    total_samples = len(y_train)

    # # クラス重みを計算
    # class_weights = {i: total_samples / count for i, count in enumerate(class_counts) if count > 0}

    baseMLPChn = args.mlp_projection_dim * 4
    # Multi layer perceptron model (pretrained network)
    model_mlp = models.MLP(baseChn=baseMLPChn, category_class_num=len(seed_vectors))

    mlp_path = os.path.join(modelPath, f'Pretrained_MLP_{baseMLPChn}')
    mlp_checkpoint_path = os.path.join(mlp_path,"model/cp.ckpt")
    mlp_checkpoint_dir = os.path.dirname(mlp_checkpoint_path)
    mlp_cp_callback = tf.keras.callbacks.ModelCheckpoint(mlp_checkpoint_path, monitor='val_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
    mlp_cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
    mlp_result_path = os.path.join(modelPath,"result/result.pkl")

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
        with open(mlp_result_path,'wb') as fp:
            pickle.dump(acc,fp)
            pickle.dump(val_acc,fp)
            pickle.dump(loss,fp)
            pickle.dump(val_loss,fp)
    else:
        # load trained parameters
        print("load models")
        model_mlp.load_weights(mlp_checkpoint_path)
        model_mlp.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
        val_loss, val_acc = model_mlp.evaluate(x_valid,y_valid)
    # train generator is to be switched to set_retrieval mode  
    train_generator = data.trainDataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num)
    x_valid, x_size_valid, y_valid = train_generator.data_generation_val()
else:
    baseMLPChn = args.mlp_projection_dim * 4
    mlp_path = os.path.join(modelPath, f'Pretrained_MLP_{baseMLPChn}')
    mlp_checkpoint_path = os.path.join(mlp_path,"model/cp.ckpt")
    mlp_checkpoint_dir = os.path.dirname(mlp_checkpoint_path)
    mlp_cp_callback = tf.keras.callbacks.ModelCheckpoint(mlp_checkpoint_path, monitor='val_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
    mlp_cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
    mlp_result_path = os.path.join(modelPath,"result/result.pkl")
    # train generator is to be switched to set_retrieval mode  
    train_generator = data.trainDataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num)
    x_valid, x_size_valid, y_valid = train_generator.data_generation_val()



# set data generator for test
# test_generator = data.testDataGenerator(year = year, cand_num = test_cand_num)
# x_test = test_generator.x
# x_size_test = test_generator.x_size
# y_test = test_generator.y
# test_batch_size = test_generator.batch_grp_num
#----------------------------

# set-matching network
model_smn = models.SMN(isCNN=False, is_TrainableMLP=True, is_set_norm=args.is_set_norm, is_cross_norm=args.is_cross_norm, num_layers=args.num_layers, num_heads=args.num_heads, baseChn=args.baseChn, baseMlp = baseMLPChn, mode=mode, calc_set_sim=calc_set_sim, rep_vec_num=rep_vec_num, seed_init = seed_vectors, is_neg_down_sample=is_neg_down_sample, use_Cvec = args.use_Cvec, is_Cvec_linear=is_Cvec_linear)

checkpoint_path = os.path.join(modelPath,"model/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_Set_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_Set_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
result_path = os.path.join(modelPath,"result/result.pkl")

if not os.path.exists(result_path):

    print("load MLP model")
    model_smn.load_weights(mlp_checkpoint_path)

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
    print("load MLP model")
    model_smn.load_weights(mlp_checkpoint_path)
    model_smn.load_weights(checkpoint_path)
#----------------------------

#---------------------------------

path = os.path.join(modelPath, "result/predict_vector.pkl")
if not os.path.exists(path):   
    # set data generator for evaluation and test (similar settings using test.pkl as train and valid)
    test_generator = data.trainDataGenerator(year = year, batch_size = batch_size, max_item_num = max_item_num)
    x_test, x_size_test, y_test, category1_test, category2_test, item_label_test = test_generator.data_generation_test()

    model_smn.compile(optimizer='adam',loss=util.Set_hinge_loss,metrics=util.Set_accuracy,run_eagerly=True)

    # test_pred_vec, test_set_score, test_category_acc, query_id, result_id, result_category2, test_ranks = model_smn.predict((x_test[:7700],x_size_test[:7700],category1_test[:7700], category2_test[:7700], item_label_test[:7700], y_test[:7700]),batch_size=batch_size, verbose=1)  
    #test_loss, test_acc = model_smn.evaluate((x_test,x_size_test),y_test,batch_size=batch_size,verbose=1) 
    test_pred_vec, gallery, replicated_set_label, query_id = model_smn.predict((x_test[:7700],x_size_test[:7700], y_test[:7700], item_label_test[:7700]),batch_size=test_batch_size, verbose=1)
    # test_pred_vec, test_set_score, query_id, result_id = model_smn.predict((x_test[:7700],x_size_test[:7700],category1_test[:7700], category2_test[:7700], item_label_test[:7700], y_test[:7700]),batch_size=test_batch_size, verbose=1)   
    
    with open(path,'wb') as fp:
        pickle.dump(test_pred_vec, fp)
        pickle.dump(gallery, fp)
        pickle.dump(y_test[:7700], fp)
        pickle.dump(replicated_set_label, fp)
        pickle.dump(query_id, fp)
        pickle.dump(category2_test[:7700],fp)
        pickle.dump(item_label_test[:7700], fp)
else:
    with open(path, 'rb') as fp:
        test_pred_vec = pickle.load(fp)
        gallery = pickle.load(fp)
        y_test = pickle.load(fp)
        replicated_set_label = pickle.load(fp)
        query_id = pickle.load(fp)
        category2_test = pickle.load(fp)
        item_label_test = pickle.load(fp)


item_retrieval = True
if item_retrieval:
    result_id = util.item_selection((gallery, test_pred_vec),(category2_test, item_label_test, replicated_set_label))
    path = os.path.join(modelPath, "result/predict_label.pkl")
    with open(path,'wb') as fp:
        pickle.dump(result_id, fp)
        pickle.dump(query_id, fp)
else:
    path = os.path.join(modelPath, "result/predict_label.pkl")
    with open(path,'rb') as fp:
        result_id = pickle.load(fp)
        query_id = pickle.load(fp)

result_id = result_id.numpy()
visualize = True
if visualize:

    # PILテキストフォントの読み込み
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    IQONPath = '/data2/yoshida/mastermatching/data/IQON/IQON3000'
    os.chdir(IQONPath)
    for batch_ind in range(0, len(query_id), test_batch_size):
        predBatchPath = os.path.join(modelPath, f"visualize/test_batch_size_{test_batch_size}/batch_{batch_ind}")
        if not os.path.exists(predBatchPath):
            os.makedirs(predBatchPath)
        vis_query_set = query_id[batch_ind : batch_ind + test_batch_size] # (Batch, nItem, 2)
        vis_result_set = result_id[batch_ind : batch_ind + test_batch_size] # (Batch, nItemCand, 2)
        images = []
        # バッチ内各集合に対して
        for query_ind in range(test_batch_size):
            query_exist = False
            target_directory_name_result_list = vis_result_set[query_ind][:,0] # 1つめの0は最近傍の意味
            target_item_result_list = vis_result_set[query_ind][:,1]
            target_directory_name = str(vis_query_set[query_ind][:,0][0])

            item_ids = vis_query_set[query_ind][:,1][np.nonzero(vis_query_set[query_ind][:,1])]
            for path in Path(".").rglob(target_directory_name):
                if path.is_dir():
                    print("Found directory of query set", path)
                    jpg_file_paths = glob.glob(os.path.join(f"{IQONPath}/{path}/", "*.jpg"))
                    
                    specified_paths = [path for path in jpg_file_paths if any(str(x) in str(path) for x in item_ids.tolist())]

                    # 画像を読み込んでリストに保存
                    for i in range(len(specified_paths)):
                        img = Image.open(specified_paths[i])
                        images.append(img)
                        query_exist = True
            if query_exist:
                blue_padding_image = Image.new("RGB", (150, 150), color=(0, 0, 255))
                images.append(blue_padding_image)
                for result_item in range(len(target_directory_name_result_list)):
                    target_directory_name_result = str(target_directory_name_result_list[result_item])
                    specified_paths = []
                    padding_pred = False
                    for path in Path(".").rglob(target_directory_name_result):
                        if path.is_dir():
                            print("Found directory of result items:", path)
                            jpg_file_paths = glob.glob(os.path.join(f"{IQONPath}/{path}/", "*.jpg"))
                            specified_paths = [path for path in jpg_file_paths if str(target_item_result_list[result_item]) in str(path)]
                            if target_item_result_list[result_item] == 0:
                                padding_pred = True

                            # 画像を読み込んでリストに保存
                    
                            if padding_pred:
                                padding_image = Image.new("RGB", (150, 150), color=(0, 0, 0)) # padding is selected for nn
                            if not specified_paths:
                                padding_image = Image.new("RGB", (150, 150), color=(255, 0, 0)) # There is no image matched (item id)                

                    # path が見つからない場合はIQONにない画像パス指定をしているとき
                    if not specified_paths: # There is no set id matched
                        padding_image = Image.new("RGB", (150, 150), color=(255, 0, 0)) # There is no image matched 
                        images.append(padding_image)
                    else: #画像の追加
                        if padding_pred:
                            images.append(padding_image)
                        else:
                            for i in range(len(specified_paths)):
                                img = Image.open(specified_paths[i])
                                images.append(img)
            if query_exist:
                concatenated_image = Image.new("RGB", (sum(img.width for img in images), max(img.height for img in images)))
                draw = ImageDraw.Draw(concatenated_image)
                x_offset = 0
                for img in images:
                    concatenated_image.paste(img, (x_offset, 0))
                    
                    if img.size == (150, 150) and img.getpixel((0, 0)) == (0, 0, 255):
                        # テキストを埋め込む位置を計算
                        text_position = (x_offset + 10, img.height // 2)
                        draw.text(text_position, "<=query images,  predict images=>", fill=(255, 255, 255), font=font)
                    x_offset += img.width

                # 連結した画像を保存
                concatenated_image.save(f"{predBatchPath}/query_{query_ind}.jpg")
            images = []





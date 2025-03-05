import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import gzip

import tensorflow as tf
from sklearn.model_selection import train_test_split

# 使用する GPU を指定（GPU 0 のみを使用）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_categories(categories_json_path):
    with open(categories_json_path, 'r') as f:
        categories = json.load(f)
    return categories

def load_styles(styles_json_path):
    with open(styles_json_path, 'r') as f:
        styles = json.load(f)
    return styles

def load_furniture_labels(furnitures_jsonl_path):
    furniture_labels = {}
    with open(furnitures_jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            furniture_id = data['furniture_id']
            category_id = data['category_id']
            style_ids = data['style_ids']
            furniture_labels[furniture_id] = {
                'category_id': category_id,
                'style_ids': style_ids
            }
    return furniture_labels

def load_image_paths_and_labels(image_dir, furniture_labels):
    image_paths = []
    category_labels = []
    style_labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            furniture_id = os.path.splitext(filename)[0]
            if furniture_id in furniture_labels:
                image_paths.append(os.path.join(image_dir, filename))
                category_labels.append(furniture_labels[furniture_id]['category_id'])
                style_ids = furniture_labels[furniture_id]['style_ids']
                if style_ids is None:
                    style_ids = []
                style_labels.append(style_ids)
    return image_paths, category_labels, style_labels

def preprocess_image(image_path, target_size=(224, 224)):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def prepare_data(image_paths, labels):
    images = []
    for path in tqdm(image_paths, desc='Loading images'):
        image = preprocess_image(path)
        images.append(image)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

#################################################################################################
# category_modelの構築と訓練
def build_category_model(num_classes, fine_tune_at=None):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # 特定の層以降を訓練可能に設定（微調整）
    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        # 全ての畳み込み層を凍結
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    # 4096 -> 2048 -> 512 -> 128 の3層のMLPを追加
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='feature_layer')(x)
    # 最終的な分類層
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

def train_category_model(X_train, y_train, X_val, y_val, num_classes, fine_tune_at=None):
    model = build_category_model(num_classes, fine_tune_at)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',  # 修正: 'sparse_categorical_crossentropy' から変更
        metrics=['accuracy']
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_category_model.weights.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint]
    )
    return model, history
#################################################################################################

#################################################################################################
# style_modelの構築と訓練
def build_style_model(num_classes, fine_tune_at=None):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # 特定の層以降を訓練可能に設定（微調整）
    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        # 全ての畳み込み層を凍結
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    # 4096 -> 2048 -> 512 -> 128 の3層のMLPを追加
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='feature_layer')(x)
    # 最終的なマルチラベル分類層（Sigmoid を使用）
    predictions = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

def train_style_model(X_train, y_train, X_val, y_val, num_classes, fine_tune_at=None):
    model = build_style_model(num_classes, fine_tune_at)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_style_model.weights.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint]
    )
    return model, history
#################################################################################################

def extract_and_save_features(model, images, labels, output_filename):
    # 特徴抽出モデルを定義
    feature_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('feature_layer').output)
    features = feature_model.predict(images, batch_size=32)
    # 特徴量とラベルをgzipで圧縮して保存
    with gzip.open(output_filename, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Features and labels have been saved as {output_filename}')

def compute_category_centers(features, labels, num_classes):
    centers = np.zeros((num_classes, features.shape[1]))
    for i in range(num_classes):
        class_features = features[labels == i]
        if class_features.shape[0] > 0:
            centers[i] = class_features.mean(axis=0)
        else:
            centers[i] = 0
    return centers

def compute_style_centers(features, style_labels, num_styles):
    centers = np.zeros((num_styles, features.shape[1]))
    for i in range(num_styles):
        # スタイルが存在する画像を選択
        class_features = features[style_labels[:, i] > 0]
        if class_features.shape[0] > 0:
            centers[i] = class_features.mean(axis=0)
        else:
            centers[i] = 0
    return centers

def main():
    # データのパスを設定
    data_dir = '/data1/yamazono/research/DeepFurniture/uncompressed_data'
    categories_json_path = os.path.join(data_dir, 'metadata', 'categories.json')
    styles_json_path = os.path.join(data_dir, 'metadata', 'styles.json')
    furnitures_jsonl_path = os.path.join(data_dir, 'metadata', 'furnitures.jsonl')
    image_dir = os.path.join(data_dir, 'furnitures')

    # カテゴリーとスタイルのマッピングを読み込む
    categories = load_categories(categories_json_path)
    styles = load_styles(styles_json_path)
    furniture_labels = load_furniture_labels(furnitures_jsonl_path)

    # 画像パスとラベルを取得
    image_paths, category_labels, style_labels = load_image_paths_and_labels(image_dir, furniture_labels)

    # カテゴリラベルを整数のインデックスに変換
    category_label_set = sorted(list(set(category_labels)))
    category_label_to_index = {label: idx for idx, label in enumerate(category_label_set)}
    category_labels = [category_label_to_index[label] for label in category_labels]
    num_category_classes = len(category_label_set)
    print(f'Number of category classes: {num_category_classes}')

    # スタイルラベルセットに「スタイルなし」を追加（0）
    style_label_set = [0] + sorted([int(k) for k in styles.keys()])
    style_label_to_index = {label: idx for idx, label in enumerate(style_label_set)}
    num_style_classes = len(style_label_set)
    print(f'Number of style classes (including "no style"): {num_style_classes}')

    # スタイルラベルを確率分布に基づく形式に変換
    style_labels_encoded = np.zeros((len(style_labels), num_style_classes))
    for i, style_ids in enumerate(style_labels):
        if style_ids is not None and len(style_ids) > 0:
            # 重複を考慮した確率計算
            unique_style_ids, counts = np.unique(style_ids, return_counts=True)
            total = counts.sum()
            for style_id, count in zip(unique_style_ids, counts):
                if style_id in style_label_to_index:
                    idx = style_label_to_index[style_id]
                    style_labels_encoded[i, idx] = count / total
        else:
            # スタイルなしクラスに均等な確率を割り当て
            style_labels_encoded[i, style_label_to_index[0]] = 1.0 / num_style_classes

    # データを読み込み、前処理
    images, category_labels = prepare_data(image_paths, category_labels)

    # データを訓練用とテスト用に分割（カテゴリ分類用）
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        images, category_labels, test_size=0.2, random_state=42, stratify=category_labels
    )

    # マルチラベルストラティファイを使用してスタイルデータを分割
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        print("iterative-stratification ライブラリがインストールされていません。インストールしてください。")
        print("pip install iterative-stratification")
        return

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in msss.split(images, style_labels_encoded):
        X_train_style, X_test_style = images[train_index], images[test_index]
        y_train_style, y_test_style = style_labels_encoded[train_index], style_labels_encoded[test_index]

    # カテゴリ分類モデルの訓練
    category_model, category_history = train_category_model(
        X_train_cat, tf.keras.utils.to_categorical(y_train_cat, num_classes=num_category_classes),  # ここを修正
        X_test_cat, tf.keras.utils.to_categorical(y_test_cat, num_classes=num_category_classes),
        num_category_classes,
        fine_tune_at=15  # 必要に応じて微調整を行う層数を指定
    )

    # スタイル分類モデルの訓練
    # style_model, style_history = train_style_model(
    #     X_train_style, y_train_style,
    #     X_test_style, y_test_style,
    #     num_style_classes,
    #     fine_tune_at=15  # 必要に応じて微調整を行う層数を指定
    # )

    # モデルの評価
    cat_loss, cat_acc = category_model.evaluate(
        X_test_cat, tf.keras.utils.to_categorical(y_test_cat, num_classes=num_category_classes), verbose=0
    )
    print(f'Category Model Test Accuracy: {cat_acc:.4f}')

    # style_loss, style_acc = style_model.evaluate(X_test_style, y_test_style, verbose=0)
    # print(f'Style Model Test Accuracy: {style_acc:.4f}')

    # 特徴量の抽出と保存
    extract_and_save_features(category_model, images, category_labels, 'yoshida_category_features.pkl.gz')
    # extract_and_save_features(style_model, images, style_labels_encoded, 'style_features.pkl.gz')

    # クラスタ中心の計算と保存
    # カテゴリのクラスタ中心を計算
    with gzip.open('yoshida_category_features.pkl.gz', 'rb') as f:
        category_data = pickle.load(f)
    category_features = category_data['features']
    category_labels_array = category_data['labels']
    category_centers = compute_category_centers(category_features, category_labels_array, num_category_classes)

    # クラスタ中心を保存
    with gzip.open('yoshida_category_centers.pkl.gz', 'wb') as f:
        pickle.dump(category_centers, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Category centers have been saved as category_centers.pkl.gz')

    # # スタイルのクラスタ中心を計算
    # with gzip.open('style_features.pkl.gz', 'rb') as f:
    #     style_data = pickle.load(f)
    # style_features = style_data['features']
    # style_labels_array = style_data['labels']
    # style_centers = compute_style_centers(style_features, style_labels_array, num_style_classes)

    # # クラスタ中心を保存
    # with gzip.open('style_centers.pkl.gz', 'wb') as f:
    #     pickle.dump(style_centers, f, protocol=pickle.HIGHEST_PROTOCOL)
    # print('Style centers have been saved as style_centers.pkl.gz')

if __name__ == '__main__':
    main()
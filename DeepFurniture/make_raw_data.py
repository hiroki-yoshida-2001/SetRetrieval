import os
import json
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pdb

# GPUの設定（必要に応じて調整）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_categories(categories_json_path):
    with open(categories_json_path, 'r') as f:
        categories = json.load(f)
    return categories

def load_furniture_labels(furnitures_jsonl_path):
    furniture_labels = {}
    with open(furnitures_jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            furniture_id = str(data['furniture_id'])  # IDを文字列に変換
            category_id = data['category_id']
            furniture_labels[furniture_id] = {
                'category_id': category_id
            }
    return furniture_labels

def load_annotations(annotations_json_path):
    with open(annotations_json_path, 'r') as f:
        annotations = json.load(f)
    return annotations  # シーンごとのデータ

def load_image_paths_and_labels(image_dir, furniture_labels):
    image_paths = []
    category_labels = []
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png')):
                furniture_id = os.path.splitext(filename)[0]
                if furniture_id in furniture_labels:
                    full_path = os.path.join(root, filename)
                    image_paths.append(full_path)
                    category_labels.append(furniture_labels[furniture_id]['category_id'])
    print(f"Found {len(image_paths)} images.")
    return image_paths, category_labels

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

def build_simple_category_model(num_classes):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True, input_tensor=inputs)
    from tensorflow.keras.models import Model
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
    
    # x = tf.keras.layers.Flatten()(base_model.output)
    
    x = model.output
    # gause_noise = np.random.randn(7*7*512, 4096)
    # x = tf.matmul(x, gause_noise)
    # pdb.set_trace()
    # x = tf.keras.layers.Dense(512, activation='gelu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    
    # # Dense層（活性化関数なし）
    # x = tf.keras.layers.Dense(128, activation=None, name='feature_layer_cat_dense')(x)
    
    # # 活性化関数を別の層として適用
    # x = tf.keras.layers.Activation('gelu', name='feature_layer_cat_activation')(x)
    
    outputs = x
    
    model = tf.keras.models.Model(inputs, outputs)
    
    # ベースモデルの層を凍結
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def build_feature_extractor(model):
    # 'feature_layer_cat_dense' 層の出力を取得（活性化前）
    feature_extractor = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.output  # 活性化前の出力
    )
    return feature_extractor


def main():
    import argparse
    import random
    import numpy as np
    import tensorflow as tf
    pdb.set_trace()
    # ログメッセージの抑制（必要に応じて）
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: 全て表示, 1: INFO非表示, 2: WARNING非表示, 3: ERRORのみ

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='家具画像ファイルが格納されているディレクトリのパス')
    parser.add_argument('--categories_json_path', type=str, required=True, help='カテゴリ情報が格納されたJSONファイルのパス')
    parser.add_argument('--furnitures_jsonl_path', type=str, required=True, help='家具ラベル情報が格納されたJSONLファイルのパス')
    parser.add_argument('--annotations_json_path', type=str, required=True, help='シーン情報が格納されたアノテーションJSONファイルのパス')
    parser.add_argument('--raw_data_path', type=str, default='deep_furniture4096.pkl', help='抽出した特徴量とラベルを保存するパス')
    parser.add_argument('--epochs', type=int, default=10, help='カテゴリモデルの学習エポック数')
    parser.add_argument('--batch_size', type=int, default=32, help='カテゴリモデルのバッチサイズ')
    args = parser.parse_args()

    # ランダムシードの設定（再現性のため）
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # メタ情報読み込み
    categories = load_categories(args.categories_json_path)
    furniture_labels = load_furniture_labels(args.furnitures_jsonl_path)
    annotations = load_annotations(args.annotations_json_path)

    # 画像パスとラベル取得
    image_paths, category_labels = load_image_paths_and_labels(args.image_dir, furniture_labels)

    if len(image_paths) == 0:
        print("Error: No images found in the specified image_dir.")
        return

    # カテゴリID→インデックス変換
    category_label_set = sorted(list(set(category_labels)))
    category_label_to_index = {lbl: i for i, lbl in enumerate(category_label_set)}
    category_labels = np.array([category_label_to_index[lbl] for lbl in category_labels])
    num_category_classes = len(category_label_set)

    # データの前処理（画像の読み込みと特徴量抽出）
    print("Preprocessing and extracting features for all images...")
    all_images, all_labels = prepare_data(image_paths, category_labels)
    print("Preprocessing completed.")

    # カテゴリモデルの構築と学習
    print("Building and training the category classification model...")
    category_model = build_simple_category_model(num_category_classes)
    category_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # category_model.fit(
    #     all_images, all_labels,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     validation_split=0.1,
    #     verbose=1
    # )
    print("Category model training completed.")

    # モデルの概要確認（デバッグ用）
    # category_model.summary()

    # 特徴量抽出（活性化関数適用前の出力）
    print("Extracting features using the trained category model...")
    feature_extractor = build_feature_extractor(category_model)
    all_features = feature_extractor.predict(all_images, batch_size=args.batch_size, verbose=1)
    print("Feature extraction completed.")

    # 特徴量の形状確認（デバッグ用）
    print("Shape of all_features:", all_features.shape)  # 期待: (num_images, 128)
    print("Sample feature vector (first image):", all_features[0])

    # シーンごとに特徴量とカテゴリIDをまとめる
    print("Aggregating scene-wise features...")
    scene_features = {}
    for annotation in tqdm(annotations, desc='Processing annotations'):
        scene = annotation.get('scene', {})
        scene_id = scene.get('sceneTaskID', None)
        if not scene_id:
            print("Warning: sceneTaskID not found in annotation.")
            continue
        instances = annotation.get('instances', [])
        item_ids = [str(instance.get('identityID')) for instance in instances if instance.get('identityID')]
        item_indices = []
        category_ids = []
        for item_id, instance in zip(item_ids, instances):
            if 'categoryID' not in instance:
                print(f"Warning: categoryID not found for item ID {item_id} in scene {scene_id}.")
                continue
            category_id = instance['categoryID']
            # アイテムIDから画像パスを構築
            jpg_path = os.path.join(args.image_dir, f"{item_id}.jpg")
            png_path = os.path.join(args.image_dir, f"{item_id}.png")
            if jpg_path in image_paths:
                idx = image_paths.index(jpg_path)
            elif png_path in image_paths:
                idx = image_paths.index(png_path)
            else:
                print(f"Warning: Item ID {item_id} not found in image directory for scene {scene_id}.")
                continue
            item_indices.append(idx)
            category_ids.append(category_id)
        if item_indices:
            # 特徴量を抽出（全画像から抽出した特徴量を使用）
            features = all_features[item_indices]
            category_ids = np.array(category_ids, dtype=int)  # shape: (n_items,)
            
            item_ids = np.array(item_ids, dtype=int) # shape: (n_items, )
            scene_features[scene_id] = {
                'features': features,            # shape: (n_items, 128)
                'item_indices': item_indices,    # list of indices
                'category_ids': category_ids ,    # array of category IDs
                'item_ids': item_ids              # array of item IDs
            }

    # 特徴量とシーンIDの保存
    print("Saving scene-wise features to raw_data.pkl...")
    # if not os.path.exists(os.path.dirname(args.raw_data_path)):
    #     os.makedirs(os.path.dirname(args.raw_data_path))
    with open(args.raw_data_path, 'wb') as f:
        pickle.dump(scene_features, f)
    print("raw_data.pkl has been created with scene-wise features, IDs, and category IDs.")


if __name__ == '__main__':
    main()
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os


class CurveDetector:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = None

    def build_model(self):
        """构建CNN模型"""
        model = models.Sequential(
            [
                # 第一层卷积
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation='relu',
                    input_shape=(self.img_size[0], self.img_size[1], 1),
                ),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                # 第二层卷积
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # 第三层卷积
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # 全连接层
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid'),
            ]
        )

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        self.model = model
        return model

    def preprocess_image(self, image_path):
        """预处理图片"""
        # 读取图片
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # 调整大小
        img = cv2.resize(img, self.img_size)

        # 归一化
        img = img / 255.0

        # 增加通道维度
        img = np.expand_dims(img, axis=-1)

        return img

    def load_dataset(self, data_dir):
        """加载训练数据集（支持curve和no_curve子目录）"""
        images = []
        labels = []

        # 定义子目录和对应的标签
        subdirs = [('curve', 1), ('no_curve', 0)]

        for subdir, label in subdirs:
            subdir_path = os.path.join(data_dir, subdir)

            # 检查子目录是否存在
            if not os.path.exists(subdir_path):
                print(f"Warning: Subdirectory '{subdir_path}' does not exist.")
                continue

            # 遍历子目录中的所有图片
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
                ):
                    img_path = os.path.join(subdir_path, filename)
                    img = self.preprocess_image(img_path)

                    if img is not None:
                        images.append(img)
                        labels.append(label)

        if len(images) == 0:
            raise ValueError(f"No valid images found in {data_dir}")

        return np.array(images), np.array(labels, dtype=np.float32)

    def train(self, train_dir, epochs=50, batch_size=16, validation_split=0.2):
        """训练模型"""
        # 加载数据
        print("Loading training data...")
        images, labels = self.load_dataset(train_dir)

        # 显示数据统计
        curve_count = np.sum(labels == 1)
        no_curve_count = np.sum(labels == 0)
        print(f"Loaded {len(images)} images for training")
        print(f"  - Curve images: {int(curve_count)}")
        print(f"  - Non-curve images: {int(no_curve_count)}")

        # 构建模型
        if self.model is None:
            self.build_model()

        # 训练模型
        print("Training model...")
        history = self.model.fit(
            images,
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
        )

        return history

    def predict_single(self, image_path):
        """预测单张图片"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load or train a model first.")

        # 预处理图片
        img = self.preprocess_image(image_path)
        if img is None:
            return 0, 0.0

        # 预测
        img_batch = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img_batch, verbose=0)

        # 返回结果（0:非曲线, 1:曲线）和置信度
        # 这里prediction[0][0]是模型输出的概率值，越接近1表示越可能是曲线
        confidence = float(prediction[0][0])
        result = 1 if confidence > 0.5 else 0

        # 如果结果是0（非曲线），置信度应该是1-confidence
        if result == 0:
            confidence = 1 - confidence

        return result, confidence

    def save_model(self, model_path='curve_detector.h5'):
        """保存模型"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")

    def load_model(self, model_path='curve_detector.h5'):
        """加载模型"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

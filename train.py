import os
import argparse
import matplotlib.pyplot as plt
from module.model import CurveDetector


def plot_training_history(history):
    """绘制训练历史图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 准确率
    axes[0, 0].plot(history.history["accuracy"], label="Training Accuracy")
    if 'val_accuracy' in history.history:
        axes[0, 0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0, 0].set_title("Model Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 损失
    axes[0, 1].plot(history.history["loss"], label="Training Loss")
    if 'val_loss' in history.history:
        axes[0, 1].plot(history.history["val_loss"], label="Validation Loss")
    axes[0, 1].set_title("Model Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 精确率
    if 'precision' in history.history:
        axes[1, 0].plot(history.history["precision"], label="Training Precision")
        if 'val_precision' in history.history:
            axes[1, 0].plot(
                history.history["val_precision"], label="Validation Precision"
            )
        axes[1, 0].set_title("Model Precision")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # 召回率
    if 'recall' in history.history:
        axes[1, 1].plot(history.history["recall"], label="Training Recall")
        if 'val_recall' in history.history:
            axes[1, 1].plot(history.history["val_recall"], label="Validation Recall")
        axes[1, 1].set_title("Model Recall")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train curve detection model")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="train/",
        help="Directory containing training images (with curve/no_curve subdirs)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="model/curve_detector.h5",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    # 检查训练目录及其子目录
    required_subdirs = ['curve', 'no_curve']
    missing_subdirs = []

    for subdir in required_subdirs:
        subdir_path = os.path.join(args.train_dir, subdir)
        if not os.path.exists(subdir_path):
            missing_subdirs.append(subdir_path)

    if missing_subdirs:
        print(f"Error: Missing required subdirectories:")
        for subdir in missing_subdirs:
            print(f"  - {subdir}")
        print("\nPlease create the directory structure:")
        print(f"  {args.train_dir}/curve/   (for curve images)")
        print(f"  {args.train_dir}/no_curve/ (for non-curve images)")
        return

    # 检查子目录中是否有图片
    has_images = False
    for subdir in required_subdirs:
        subdir_path = os.path.join(args.train_dir, subdir)
        if os.path.exists(subdir_path):
            images = [
                f
                for f in os.listdir(subdir_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
            ]
            if images:
                has_images = True
                break

    if not has_images:
        print(
            f"Error: No images found in {args.train_dir}/curve/ or {args.train_dir}/no_curve/"
        )
        print(
            "Please add curve images to 'curve/' and non-curve images to 'no_curve/' subdirectories"
        )
        return

    # 初始化检测器
    detector = CurveDetector(img_size=(128, 128))

    # 训练模型
    print("=" * 50)
    print("Starting Curve Detection Model Training")
    print("=" * 50)

    try:
        # 确保model目录存在
        os.makedirs("model", exist_ok=True)

        history = detector.train(
            train_dir=args.train_dir, epochs=args.epochs, batch_size=args.batch_size
        )

        # 保存模型
        detector.save_model(args.model_save_path)

        # 绘制训练历史
        plot_training_history(history)

        print("\nTraining completed successfully!")
        print(f"Model saved as: {args.model_save_path}")
        print(f"Training history plot saved as: training_history.png")

    except Exception as e:
        print(f"Error during training: {str(e)}")


if __name__ == "__main__":
    main()

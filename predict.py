import os
import argparse
import shutil
from datetime import datetime
from module.model import CurveDetector
import cv2
import numpy as np


def analyze_results(results, test_dir):
    """分析预测结果并生成报告"""
    total = len(results)

    # 修改这里：results现在有5个值，我们需要第二个值(is_curve)
    curve_count = sum(1 for _, is_curve, _, _, _ in results if is_curve)
    non_curve_count = total - curve_count

    print("\n" + "=" * 50)
    print("PREDICTION ANALYSIS REPORT")
    print("=" * 50)
    print(f"Total images processed: {total}")
    print(f"Curve images detected: {curve_count} ({curve_count/total*100:.1f}%)")
    print(
        f"Non-curve images detected: {non_curve_count} ({non_curve_count/total*100:.1f}%)"
    )

    # 计算平均置信度
    if total > 0:
        avg_confidence = sum(confidence for _, _, confidence, _, _ in results) / total
        print(f"Average confidence: {avg_confidence:.2%}")

    # 找到最高和最低置信度的图片
    if results:
        # 修改这里：获取置信度（第三个值）
        max_conf = max(results, key=lambda x: x[2])
        min_conf = min(results, key=lambda x: x[2])

        print(
            f"\nHighest confidence: {max_conf[0]} - {max_conf[2]:.2%} (Curve: {max_conf[1]})"
        )
        print(
            f"Lowest confidence: {min_conf[0]} - {min_conf[2]:.2%} (Curve: {min_conf[1]})"
        )

    print("\nSummary:")
    if curve_count / total > 0.7:
        print("✓ Most images contain curves")
    elif non_curve_count / total > 0.7:
        print("✗ Most images do not contain curves")
    else:
        print("∼ Mixed content - both curves and non-curves detected")

    # 建议
    print("\nRecommendations:")
    if non_curve_count > 0:
        print(f"- Review {non_curve_count} non-curve images in 'goal/false/' directory")
    if curve_count > 0:
        print(f"- {curve_count} curve images have been saved to 'goal/true/' directory")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Predict curves in images")
    parser.add_argument(
        "--test_dir", type=str, default="test/", help="Directory containing test images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/curve_detector.h5",
        help="Path to trained model",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for prediction",
    )

    args = parser.parse_args()

    # 检查测试目录
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' does not exist.")
        print("Please create a 'test/' directory and add images to test.")
        return

    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        print("Please train the model first using: python train.py")
        return

    # 创建目标目录
    os.makedirs("goal/true", exist_ok=True)
    os.makedirs("goal/false", exist_ok=True)
    os.makedirs("log", exist_ok=True)  # 创建log文件夹

    # 加载模型
    print("Loading model...")
    detector = CurveDetector()
    detector.load_model(args.model_path)

    # 获取测试图片（支持test目录下的所有子目录）
    test_images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    # 遍历test目录及其所有子目录
    for root, dirs, files in os.walk(args.test_dir):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                # 获取完整路径
                full_path = os.path.join(root, filename)
                # 获取相对于test_dir的路径，用于显示
                rel_path = os.path.relpath(full_path, args.test_dir)
                test_images.append((rel_path, full_path))

    if not test_images:
        print(f"No images found in {args.test_dir} or its subdirectories")
        return

    print(f"Found {len(test_images)} images for prediction")
    print("Starting prediction...")

    results = []
    for i, (rel_filename, full_path) in enumerate(test_images, 1):
        # 预测
        is_curve, confidence = detector.predict_single(full_path)

        # 决定最终分类（基于置信度阈值）
        final_prediction = 1 if confidence > args.confidence_threshold else 0

        # 复制图片到相应目录
        if final_prediction == 1:  # 曲线
            dest_dir = "goal/true"
            curve_status = "Curve"
        else:  # 非曲线
            dest_dir = "goal/false"
            curve_status = "Non-curve"

        # 生成新的文件名（包含置信度信息）
        # 将路径中的分隔符替换为下划线，避免创建子目录
        safe_filename = (
            rel_filename.replace(os.path.sep, '_').replace('/', '_').replace('\\', '_')
        )
        name, ext = os.path.splitext(safe_filename)
        new_filename = f"{name}_conf{confidence:.2f}{ext}"
        dest_path = os.path.join(dest_dir, new_filename)

        # 复制文件
        shutil.copy2(full_path, dest_path)

        # 修改这里：现在有5个值
        # (原始文件名, is_curve, 置信度, 曲线类型, 保存后的文件名)
        results.append(
            (rel_filename, final_prediction, confidence, curve_status, new_filename)
        )

        print(
            f"[{i}/{len(test_images)}] {rel_filename}: {curve_status} (Confidence: {confidence:.2%})"
        )

    # 生成分析报告
    analyze_results(results, args.test_dir)

    # 保存置信度信息到CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"log/confidence_report_{timestamp}.csv"

    with open(csv_file, "w", encoding="utf-8") as f:
        # 写入CSV表头
        f.write(
            "original_filename,is_curve,confidence,curve_type,saved_filename,threshold\n"
        )

        # 写入每条记录
        for filename, is_curve, confidence, curve_type, saved_filename in results:
            f.write(
                f"{filename},{is_curve},{confidence:.6f},{curve_type},{saved_filename},{args.confidence_threshold}\n"
            )

        # 写入统计信息
        total = len(results)
        curve_count = sum(1 for _, is_curve, _, _, _ in results if is_curve)
        non_curve_count = total - curve_count
        avg_confidence = (
            sum(confidence for _, _, confidence, _, _ in results) / total
            if total > 0
            else 0
        )

        f.write("\n# Statistics\n")
        f.write(f"# total_images,{total}\n")
        f.write(f"# curve_images,{curve_count}\n")
        f.write(f"# non_curve_images,{non_curve_count}\n")
        f.write(f"# curve_percentage,{curve_count/total*100:.2f}%\n")
        f.write(f"# non_curve_percentage,{non_curve_count/total*100:.2f}%\n")
        f.write(f"# average_confidence,{avg_confidence:.6f}\n")
        f.write(f"# confidence_threshold,{args.confidence_threshold}\n")
        f.write(f"# test_directory,{args.test_dir}\n")
        f.write(f"# model_path,{args.model_path}\n")
        f.write(f"# timestamp,{timestamp}\n")

    print(f"\n置信度报告已保存到: {csv_file}")
    print("提示: 可用Excel直接打开此CSV文件进行数据分析")
    print("图片已分类保存到 'goal/' 目录")


if __name__ == "__main__":
    main()

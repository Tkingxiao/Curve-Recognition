import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image
import threading
from pathlib import Path


class ImageConverter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("图片转换器 - 转PNG格式")
        self.root.geometry("500x300")

        # 设置窗口图标（如果有的话）
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        self.setup_ui()

    def setup_ui(self):
        # 标题
        title_label = tk.Label(
            self.root, text="图片格式转换器", font=("Arial", 20, "bold")
        )
        title_label.pack(pady=20)

        # 说明文本
        desc_label = tk.Label(
            self.root,
            text="支持格式: JPG, JPEG, WEBP, PNG, BMP, GIF, TIFF\n转换为PNG格式，不保留原文件",
            font=("Arial", 10),
        )
        desc_label.pack(pady=10)

        # 选择文件夹按钮
        self.select_btn = tk.Button(
            self.root,
            text="选择文件夹并开始转换",
            command=self.start_conversion,
            font=("Arial", 12),
            height=2,
            width=20,
        )
        self.select_btn.pack(pady=20)

        # 进度标签
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.progress_label.pack(pady=10)

        # 状态文本
        self.status_text = tk.Text(self.root, height=8, width=50)
        self.status_text.pack(pady=10)

        # 滚动条
        scrollbar = tk.Scrollbar(self.root)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)

    def start_conversion(self):
        # 禁用按钮防止重复点击
        self.select_btn.config(state=tk.DISABLED)

        # 在新线程中执行转换，避免界面卡顿
        thread = threading.Thread(target=self.convert_images)
        thread.daemon = True
        thread.start()

    def convert_images(self):
        # 弹出文件夹选择对话框
        folder_path = filedialog.askdirectory(title="选择包含图片的文件夹")

        if not folder_path:
            self.select_btn.config(state=tk.NORMAL)
            return

        # 支持的图片格式
        supported_formats = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".gif",
            ".tiff",
            ".tif",
        ]

        # 统计信息
        total_files = 0
        converted_files = 0
        failed_files = []

        # 更新状态
        self.update_status(f"正在扫描文件夹: {folder_path}\n")

        # 遍历文件夹
        for root_dir, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root_dir, file)
                file_ext = Path(file).suffix.lower()

                # 检查是否是支持的图片格式
                if file_ext in supported_formats:
                    total_files += 1

                    try:
                        # 打开图片
                        img = Image.open(file_path)

                        # 如果是PNG格式且不需要转换，则跳过
                        if file_ext == ".png":
                            self.update_status(f"跳过PNG文件: {file}\n")
                            continue

                        # 生成新文件名（相同路径，不同扩展名）
                        new_file_path = os.path.splitext(file_path)[0] + ".png"

                        # 转换为PNG
                        if img.mode in ("RGBA", "LA") or (
                            img.mode == "P" and "transparency" in img.info
                        ):
                            # 保留透明度
                            img.save(new_file_path, "PNG")
                        else:
                            # 转换为RGB模式（无透明度）
                            rgb_img = img.convert("RGB")
                            rgb_img.save(new_file_path, "PNG")

                        # 删除原文件
                        os.remove(file_path)

                        converted_files += 1
                        self.update_status(
                            f"✓ 已转换: {file} -> {Path(file).stem}.png\n"
                        )

                    except Exception as e:
                        failed_files.append((file, str(e)))
                        self.update_status(f"✗ 转换失败: {file} - {str(e)}\n")

        # 显示统计信息
        self.update_status("\n" + "=" * 50 + "\n")
        self.update_status(f"转换完成！\n")
        self.update_status(f"总图片数: {total_files}\n")
        self.update_status(f"成功转换: {converted_files}\n")
        self.update_status(f"失败: {len(failed_files)}\n")

        if failed_files:
            self.update_status("\n失败的文件:\n")
            for file, error in failed_files:
                self.update_status(f"  {file}: {error}\n")

        # 重新启用按钮
        self.root.after(0, lambda: self.select_btn.config(state=tk.NORMAL))

        # 弹出完成提示
        messagebox.showinfo(
            "转换完成",
            f"转换完成！\n成功转换 {converted_files} 个文件\n失败 {len(failed_files)} 个文件",
        )

    def update_status(self, message):
        # 在UI线程中更新状态文本
        self.root.after(0, self._update_status_text, message)

    def _update_status_text(self, message):
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
        self.status_text.update()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # 检查必要的库
    try:
        converter = ImageConverter()
        converter.run()
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("\n请安装所需的库:")
        print("pip install pillow")
        input("按回车键退出...")

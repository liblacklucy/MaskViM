import os
import shutil


def collect_images(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)
                print(f'图片已复制：{src_path} -> {dst_path}')


# 使用示例
source_directory = '/data/ljh/data/datasets/cityscapes/gtFine/val'  # 替换为源目录路径
destination_directory = '/data/ljh/data/datasets/cityscapes-c/gtFine/val'  # 替换为目标目录路径
collect_images(source_directory, destination_directory)

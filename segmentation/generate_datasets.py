import os
import shutil
from random import sample

# 源图片文件夹和目标图片文件夹路径
source_images_folder = '/data/ljh/data/datasets/mapillary/validation/images'
target_images_folder = '/data/ljh/data/datasets/mapillary/validation/images_1_2000'
# source_images_folder = '/data/ljh/data/datasets/gta/images/valid'
# target_images_folder = '/data/ljh/data/datasets/gta/images/valid_1_80'
# source_images_folder = '/data/ljh/data/datasets/synthia/RGB/val'
# target_images_folder = '/data/ljh/data/datasets/synthia/RGB/val_1_80'

# 源标签文件夹和目标标签文件夹路径
source_labels_folder = '/data/ljh/data/datasets/mapillary/labels_detectron2/validation'
target_labels_folder = '/data/ljh/data/datasets/mapillary/labels_detectron2/validation_1_2000'
# source_labels_folder = '/data/ljh/data/datasets/gta/labels_detectron2/valid'
# target_labels_folder = '/data/ljh/data/datasets/gta/labels_detectron2/valid_1_80'
# source_labels_folder = '/data/ljh/data/datasets/synthia/labels_detectron2/val'
# target_labels_folder = '/data/ljh/data/datasets/synthia/labels_detectron2/val_1_80'

# 如果目标文件夹不存在，则创建它们
if not os.path.exists(target_images_folder):
    os.makedirs(target_images_folder)
if not os.path.exists(target_labels_folder):
    os.makedirs(target_labels_folder)

# 获取源图片文件夹中所有图片的列表
all_images = [file for file in os.listdir(source_images_folder) if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# 随机选择一半的图片
selected_images = sample(all_images, len(all_images) // 2000)

# 将选中的图片复制到目标图片文件夹
for image in selected_images:
    shutil.copy(os.path.join(source_images_folder, image), os.path.join(target_images_folder, image))

# 根据选中的图片名称，找到对应的标签并复制到目标标签文件夹
for image in selected_images:
    # 假设标签文件与图片文件同名，但扩展名不同
    label_file = image.rsplit('.', 1)[0] + '.png'  # 假设标签文件的扩展名为.txt
    shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(target_labels_folder, label_file))

print(f'已将选中的图片复制到 {target_images_folder}')
print(f'已将对应的标签复制到 {target_labels_folder}')

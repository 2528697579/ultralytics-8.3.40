#合并原本数据集train+增强数据集train_aug
import os
import shutil
from tqdm import tqdm

# 定义路径
original_images_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train/images"
original_labels_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train/labels"

augmented_images_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train_aug/images"
augmented_labels_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train_aug/labels"

combined_images_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train_combined/images"
combined_labels_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train_combined/labels"

# 创建组合数据集目录
os.makedirs(combined_images_dir, exist_ok=True)
os.makedirs(combined_labels_dir, exist_ok=True)

# 复制原始图像和标签
original_image_files = [f for f in os.listdir(original_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in tqdm(original_image_files, desc="复制原始图像"):
    shutil.copyfile(os.path.join(original_images_dir, image_file), os.path.join(combined_images_dir, image_file))
    label_file = os.path.splitext(image_file)[0] + '.txt'
    shutil.copyfile(os.path.join(original_labels_dir, label_file), os.path.join(combined_labels_dir, label_file))

# 复制增强图像和标签
augmented_image_files = [f for f in os.listdir(augmented_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in tqdm(augmented_image_files, desc="复制增强图像"):
    # 为避免文件名冲突，可以在增强图像名前加前缀，如 "aug_"
    new_image_file = "aug_" + image_file
    shutil.copyfile(os.path.join(augmented_images_dir, image_file), os.path.join(combined_images_dir, new_image_file))
    label_file = os.path.splitext(image_file)[0] + '.txt'
    new_label_file = "aug_" + label_file
    shutil.copyfile(os.path.join(augmented_labels_dir, label_file), os.path.join(combined_labels_dir, new_label_file))
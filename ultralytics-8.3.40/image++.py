import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shutil
from tqdm import tqdm

# 定义路径
original_images_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train/images"
original_labels_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train/labels"

augmented_images_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train_aug/images"
augmented_labels_dir = "C:/Users/25286/Desktop/ultralytics-8.3.40/datasets/Unified-DataSet/train_aug/labels"

# 创建增强后目录
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)

# 定义Albumentations增强流水线
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 水平翻转
    A.VerticalFlip(p=0.2),  # 垂直翻转
    A.Rotate(limit=15, p=0.5),  # 随机旋转 ±15 度
    A.RandomBrightnessContrast(p=0.5),  # 随机亮度和对比度调整
    A.HueSaturationValue(p=0.3),  # 随机调整色调、饱和度和明度
    A.GaussNoise(p=0.3),  # 增加高斯噪音
    A.Blur(blur_limit=3, p=0.2),  # 模糊处理
    A.CLAHE(p=0.2)  # 自适应直方图均衡
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def augment_image(image_path, label_path, output_image_path, output_label_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    height, width, _ = image.shape

    # 读取标签（YOLO格式）
    with open(label_path, 'r') as f:
        labels = f.readlines()

    bboxes = []
    class_labels = []
    for label in labels:
        parts = label.strip().split()
        if len(parts) < 5:
            continue  # 跳过格式不正确的标签
        class_id = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        bbox_width = float(parts[3])
        bbox_height = float(parts[4])
        # YOLO格式的bbox为相对坐标，因此无需转换
        bboxes.append([x_center, y_center, bbox_width, bbox_height])
        class_labels.append(class_id)

    # 如果没有BBox，直接复制图像
    if len(bboxes) == 0:
        shutil.copyfile(image_path, output_image_path)
        open(output_label_path, 'w').close()  # 创建空标签文件
        return

    # 应用增强
    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_class_labels = augmented['class_labels']
    except Exception as e:
        print(f"增强失败: {image_path}，错误: {e}")
        return

    # 保存增强后的图像
    cv2.imwrite(output_image_path, augmented_image)

    # 保存增强后的标签
    with open(output_label_path, 'w') as f:
        for bbox, class_id in zip(augmented_bboxes, augmented_class_labels):
            if len(bbox) != 4:
                continue  # 跳过格式不正确的BBox
            x_center, y_center, bbox_width, bbox_height = bbox
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


# 获取所有图像文件
image_files = [f for f in os.listdir(original_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 遍历所有图像并进行增强
for image_file in tqdm(image_files, desc="数据增强中"):
    image_path = os.path.join(original_images_dir, image_file)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(original_labels_dir, label_file)

    output_image_path = os.path.join(augmented_images_dir, image_file)
    output_label_path = os.path.join(augmented_labels_dir, label_file)

    augment_image(image_path, label_path, output_image_path, output_label_path)

print("数据增强完成！")
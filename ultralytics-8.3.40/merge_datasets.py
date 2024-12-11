# 合并xyu和k2000的数据集
import os
import xml.etree.ElementTree as ET
import shutil
import random

# ==================== 配置部分 ====================

# 统一的类别列表
CLASSES = ["helmet", "without_helmet", "two_wheeler"]

# 类别名称到类别 ID 的映射（数据集A和数据集B）
class_mapping_A = {
    "helmet": 0,
    "no_helmet": 1,         # 统一为 without_helmet
    "without_helmet": 1,    # 统一为 without_helmet
    "electric": 2,          # 统一为 two_wheeler
    "two_wheeler": 2        # 统一为 two_wheeler
}

class_mapping_B = {
    "helmet": 0,
    "without_helmet": 1,
    "two_wheeler": 2
}

# 数据集A和B的路径（请根据实际情况修改）
DATASET_A_XML_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\xyu_buy\Annotations"          # 数据集A的XML注释目录
DATASET_A_IMAGES_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\xyu_buy\JPEGImages"        # 数据集A的图片目录
DATASET_B_TRAIN_LABELS_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\KY0002\helmetdataset\train\labels"  # 数据集B的训练YOLO注释目录
DATASET_B_VAL_LABELS_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\KY0002\helmetdataset\val\labels"      # 数据集B的验证YOLO注释目录
DATASET_B_TRAIN_IMAGES_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\KY0002\helmetdataset\train\images"  # 数据集B的训练图片目录
DATASET_B_VAL_IMAGES_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\KY0002\helmetdataset\val\images"      # 数据集B的验证图片目录

# 定义中间和输出目录
CONVERTED_A_YOLO_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\Unified-DataSet\labels\converted_A"  # 数据集A转换后的YOLO注释目录
MERGED_YOLO_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\Unified-DataSet\labels\merged"           # 合并后的YOLO注释目录
UNIFIED_IMAGES_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\Unified-DataSet\images\full"           # 合并后的图片目录

# 最终统一的labels和images目录
FINAL_LABELS_DIR = MERGED_YOLO_DIR
FINAL_IMAGES_DIR = UNIFIED_IMAGES_DIR

# 定义输出基础目录
OUTPUT_BASE_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\Unified-DataSet"

# 定义划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 随机种子（确保可重复性）
RANDOM_SEED = 42

# ==================== 功能函数部分 ====================

def convert_bbox_to_yolo(size, bbox):
    """
    将边界框坐标转换为 YOLO 格式。

    Args:
        size (tuple): 图像尺寸 (宽度, 高度)。
        bbox (tuple): 边界框坐标 (xmin, ymin, xmax, ymax)。

    Returns:
        tuple: (中心_x, 中心_y, 宽度, 高度) 归一化后坐标。
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin

    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh

    return (x_center, y_center, width, height)

def convert_voc_to_yolo(xml_file, class_mapping, output_dir, prefix=""):
    """
    将一个 XML 注释文件转换为 YOLO 格式的 txt 文件，并根据需要添加前缀。

    Args:
        xml_file (str): XML 文件路径。
        class_mapping (dict): 类别名称到类别 ID 的映射。
        output_dir (str): 输出 txt 文件的目录。
        prefix (str): 可选的文件名前缀。
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {xml_file}: {e}")
        return

    # 获取图像尺寸
    size = root.find('size')
    if size is None:
        print(f"文件 {xml_file} 中找不到 <size> 标签，跳过。")
        return
    try:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    except (AttributeError, ValueError):
        print(f"文件 {xml_file} 中的尺寸信息有误，跳过。")
        return

    yolo_annotations = []

    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()

        # 统一类别名称
        if name in ['no_helmet', 'without_helmet']:
            unified_name = 'without_helmet'
        elif name in ['electric', 'two_wheeler']:
            unified_name = 'two_wheeler'
        elif name == 'helmet':
            unified_name = 'helmet'
        else:
            print(f"未知类别名称 '{name}' 在文件 {xml_file} 中，跳过。")
            continue

        if unified_name not in class_mapping:
            print(f"类别名称 '{unified_name}' 不在类别列表中，跳过。")
            continue
        class_id = class_mapping[unified_name]

        bndbox = obj.find('bndbox')
        if bndbox is None:
            print(f"文件 {xml_file} 中对象 '{name}' 缺少 <bndbox> 标签，跳过。")
            continue

        try:
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
        except (AttributeError, ValueError) as e:
            print(f"文件 {xml_file} 中对象 '{name}' 的边界框坐标错误：{e}，跳过。")
            continue

        # 检查边界框的有效性
        if xmax <= xmin or ymax <= ymin:
            print(f"文件 {xml_file} 中对象 '{name}' 的边界框无效，跳过。")
            continue

        yolo_bbox = convert_bbox_to_yolo((width, height), (xmin, ymin, xmax, ymax))
        yolo_annotations.append(f"{class_id} " + " ".join([f"{coord:.6f}" for coord in yolo_bbox]))

    if not yolo_annotations:
        print(f"文件 {xml_file} 中没有有效的对象，跳过保存。")
        return

    # 定义 txt 文件的保存路径，并添加前缀（如果有）
    basename = os.path.basename(xml_file)
    txt_filename = prefix + os.path.splitext(basename)[0] + '.txt'
    txt_file = os.path.join(output_dir, txt_filename)

    # 写入到 txt 文件
    with open(txt_file, 'w') as f:
        for anno in yolo_annotations:
            f.write(anno + '\n')
    print(f"已转换并保存 {txt_file}")

def convert_dataset_A():
    """
    转换数据集A的所有XML注释为YOLO格式。
    """
    print("正在转换数据集A的XML注释为YOLO格式...")
    if not os.path.exists(CONVERTED_A_YOLO_DIR):
        os.makedirs(CONVERTED_A_YOLO_DIR)

    for filename in os.listdir(DATASET_A_XML_DIR):
        if not filename.endswith(".xml"):
            continue
        xml_path = os.path.join(DATASET_A_XML_DIR, filename)
        convert_voc_to_yolo(xml_path, class_mapping_A, CONVERTED_A_YOLO_DIR)
    print("数据集A的转换完成。\n")

def copy_dataset_B_labels(prefix="B_"):
    """
    将数据集B的标签从train和val文件夹复制到merged目录，并重命名。

    Args:
        prefix (str): 标签文件的前缀，默认是 "B_"
    """
    print("正在复制数据集B的YOLO注释到merged目录...")
    if not os.path.exists(MERGED_YOLO_DIR):
        os.makedirs(MERGED_YOLO_DIR)

    # 复制train标签并重命名
    train_labels = os.listdir(DATASET_B_TRAIN_LABELS_DIR)
    for filename in train_labels:
        if not filename.endswith(".txt"):
            continue
        src_path = os.path.join(DATASET_B_TRAIN_LABELS_DIR, filename)
        new_filename = f"{prefix}{filename}"
        dest_path = os.path.join(MERGED_YOLO_DIR, new_filename)
        with open(src_path, 'r') as src_file:
            lines = src_file.readlines()
        with open(dest_path, 'w') as dest_file:
            dest_file.writelines(lines)

    # 复制val标签并重命名
    val_labels = os.listdir(DATASET_B_VAL_LABELS_DIR)
    for filename in val_labels:
        if not filename.endswith(".txt"):
            continue
        src_path = os.path.join(DATASET_B_VAL_LABELS_DIR, filename)
        new_filename = f"{prefix}{filename}"
        dest_path = os.path.join(MERGED_YOLO_DIR, new_filename)
        with open(src_path, 'r') as src_file:
            lines = src_file.readlines()
        with open(dest_path, 'w') as dest_file:
            dest_file.writelines(lines)
    print("数据集B的YOLO注释复制完成。\n")

def merge_annotations():
    """
    合并数据集A和数据集B的YOLO注释到merged目录。
    """
    print("正在合并数据集A和数据集B的YOLO注释...")
    # 首先复制数据集B的标签（已重命名）
    copy_dataset_B_labels()

    # 然后合并数据集A的标签
    for filename in os.listdir(CONVERTED_A_YOLO_DIR):
        if not filename.endswith(".txt"):
            continue
        merged_txt_path = os.path.join(MERGED_YOLO_DIR, filename)
        converted_a_txt_path = os.path.join(CONVERTED_A_YOLO_DIR, filename)
        if os.path.exists(converted_a_txt_path):
            with open(converted_a_txt_path, 'r') as a_file:
                a_lines = a_file.readlines()
            # 如果标签文件来自数据集A和B，您可能需要合并类别，而不是简单地追加
            # 这里假设文件名不冲突（因为数据集B的标签已被重命名）
            with open(merged_txt_path, 'a') as merged_file:
                merged_file.writelines(a_lines)
    print("数据集A和数据集B的YOLO注释合并完成。\n")

def collect_all_images(prefix="B_"):
    """
    将数据集A和数据集B的所有图片复制到统一的images目录，并为数据集B的图片添加前缀以避免名称冲突。

    Args:
        prefix (str): 数据集B的图片前缀，默认是 "B_"
    """
    print("正在复制所有图片到统一的images目录...")
    if not os.path.exists(UNIFIED_IMAGES_DIR):
        os.makedirs(UNIFIED_IMAGES_DIR)

    # 复制数据集A的图片
    for filename in os.listdir(DATASET_A_IMAGES_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        src_path = os.path.join(DATASET_A_IMAGES_DIR, filename)
        dest_path = os.path.join(UNIFIED_IMAGES_DIR, filename)
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

    # 复制数据集B的train图片并重命名
    for filename in os.listdir(DATASET_B_TRAIN_IMAGES_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        src_path = os.path.join(DATASET_B_TRAIN_IMAGES_DIR, filename)
        new_filename = f"{prefix}{filename}"
        dest_path = os.path.join(UNIFIED_IMAGES_DIR, new_filename)
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

    # 复制数据集B的val图片并重命名
    for filename in os.listdir(DATASET_B_VAL_IMAGES_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        src_path = os.path.join(DATASET_B_VAL_IMAGES_DIR, filename)
        new_filename = f"{prefix}{filename}"
        dest_path = os.path.join(UNIFIED_IMAGES_DIR, new_filename)
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

    print("所有图片已复制到统一的images目录。\n")

def get_image_filenames():
    """
    获取统一图片目录中的所有图片文件名（不含扩展名）。
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    filenames = []
    for f in os.listdir(UNIFIED_IMAGES_DIR):
        if f.lower().endswith(image_extensions):
            filenames.append(os.path.splitext(f)[0])
    return filenames

def split_dataset(filenames, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    按照比例划分数据集为train、val和test。

    Args:
        filenames (list): 所有图片的文件名（不含扩展名）。
        train_ratio (float): 训练集比例。
        val_ratio (float): 验证集比例。
        test_ratio (float): 测试集比例。

    Returns:
        dict: 包含划分后的train、val和test列表。
    """
    total = len(filenames)
    random.shuffle(filenames)

    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_files = filenames[:train_end]
    val_files = filenames[train_end:val_end]
    test_files = filenames[val_end:]

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

def write_split_files(splits, output_dir):
    """
    将划分结果写入对应的文件夹。

    Args:
        splits (dict): 包含train、val和test列表的字典。
        output_dir (str): 最终数据集的根目录。
    """
    for split, files in splits.items():
        split_images_dir = os.path.join(output_dir, split, "images")
        split_labels_dir = os.path.join(output_dir, split, "labels")

        try:
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)
        except Exception as e:
            print(f"无法创建目录 {split_images_dir} 或 {split_labels_dir}，错误信息：{e}")
            continue

        for filename in files:
            # 复制图片
            src_image = None
            # 尝试查找加前缀的数据集B的图片
            potential_filenames = [
                f"{filename}.jpg",
                f"{filename}.jpeg",
                f"{filename}.png",
                f"{filename}.bmp",
                f"B_{filename}.jpg",
                f"B_{filename}.jpeg",
                f"B_{filename}.png",
                f"B_{filename}.bmp"
            ]
            for fname in potential_filenames:
                potential_path = os.path.join(UNIFIED_IMAGES_DIR, fname)
                if os.path.exists(potential_path):
                    src_image = potential_path
                    break
            if src_image is None:
                print(f"图片 {filename} 未找到，跳过。")
                continue

            try:
                shutil.copy2(src_image, os.path.join(split_images_dir, os.path.basename(src_image)))
            except Exception as e:
                print(f"无法复制图片 {src_image} 到 {split_images_dir}，错误信息：{e}")
                continue

            # 复制标签
            src_label = os.path.join(FINAL_LABELS_DIR, f"{filename}.txt")
            # 尝试查找加前缀的数据集B的标签
            if not os.path.exists(src_label):
                src_label = os.path.join(FINAL_LABELS_DIR, f"B_{filename}.txt")
            dest_label = os.path.join(split_labels_dir, os.path.basename(src_label))
            if os.path.exists(src_label):
                try:
                    shutil.copy2(src_label, dest_label)
                except Exception as e:
                    print(f"无法复制标签 {src_label} 到 {split_labels_dir}，错误信息：{e}")
                    continue
            else:
                # 如果没有标签文件，可以选择创建一个空文件或跳过
                # 这里选择跳过
                print(f"标签文件 {os.path.basename(src_label)} 不存在，跳过。")
                continue

        print(f"已将 {len(files)} 张图片和标签复制到 {split} 集。\n")

def create_classes_file(output_dir, classes):
    """
    创建统一的 classes.txt 文件。

    Args:
        output_dir (str): classes.txt 文件的保存目录。
        classes (list): 类别名称列表。
    """
    classes_txt_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_txt_path, 'w') as f:
        for cls in classes:
            f.write(cls + '\n')
    print(f"已创建统一的 classes.txt 文件在 {classes_txt_path}\n")

def main():
    """
    主函数，完成数据集A和数据集B的转换、合并与划分。
    """
    # 设置随机种子
    random.seed(RANDOM_SEED)

    # 步骤1：转换数据集A的XML注释为YOLO格式
    convert_dataset_A()

    # 步骤2：合并数据集A和数据集B的YOLO注释
    merge_annotations()

    # 步骤3：复制所有图片到统一的images目录，并重命名数据集B的图片
    collect_all_images()

    # 步骤4：获取所有图片文件名
    all_filenames = get_image_filenames()
    print(f"总共收集到 {len(all_filenames)} 张图片。\n")

    # 步骤5：划分数据集
    splits = split_dataset(all_filenames, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)
    print(f"数据集划分为：训练集 {len(splits['train'])}，验证集 {len(splits['val'])}，测试集 {len(splits['test'])}。\n")

    # 步骤6：复制图片和标签到对应的train、val和test文件夹
    write_split_files(splits, OUTPUT_BASE_DIR)

    # 步骤7：创建统一的 classes.txt 文件
    create_classes_file(os.path.join(OUTPUT_BASE_DIR, "train", "labels"), CLASSES)

    print("所有步骤已完成。请检查最终的YOLO注释和图片目录。")

if __name__ == "__main__":
    main()
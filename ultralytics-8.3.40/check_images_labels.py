#快速查看图片与labels是否对应，有哪些图片是没有labels或者有那些labels没有图片？
import os

def get_filenames(directory, extensions):
    """
    获取指定目录中所有指定扩展名的文件名（不含扩展名）。

    Args:
        directory (str): 要扫描的目录路径。
        extensions (tuple): 允许的文件扩展名。

    Returns:
        set: 文件名集合（不含扩展名）。
    """
    filenames = set()
    for filename in os.listdir(directory):
        if filename.lower().endswith(extensions):
            basename = os.path.splitext(filename)[0]
            filenames.add(basename)
    return filenames

def check_correspondence(images_dir, labels_dir):
    """
    检查图片目录和标签目录中的文件是否一一对应。

    Args:
        images_dir (str): 图片目录路径。
        labels_dir (str): 标签目录路径。

    Returns:
        tuple: 两个集合，分别包含有图片但缺少标签的文件名和有标签但缺少图片的文件名。
    """
    # 定义图片和标签的文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    label_extensions = ('.txt',)  # YOLO格式标签通常是txt文件

    # 获取图片和标签的文件名集合
    image_filenames = get_filenames(images_dir, image_extensions)
    label_filenames = get_filenames(labels_dir, label_extensions)

    # 找出有图片但缺少标签的文件名
    images_without_labels = image_filenames - label_filenames

    # 找出有标签但缺少图片的文件名
    labels_without_images = label_filenames - image_filenames

    return images_without_labels, labels_without_images

def main():
    # 请根据实际情况修改以下路径
    IMAGES_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\Unified-DataSet\val\images"  # 图片目录
    LABELS_DIR = r"C:\Users\25286\Desktop\ultralytics-8.3.40\datasets\Unified-DataSet\val\labels"  # 标签目录

    # 检查对应关系
    images_missing_labels, labels_missing_images = check_correspondence(IMAGES_DIR, LABELS_DIR)

    # 输出结果
    if images_missing_labels:
        print(f"共有 {len(images_missing_labels)} 张图片缺少对应的标签文件：")
        for fname in sorted(images_missing_labels):
            print(f"  - {fname}")
    else:
        print("所有图片都有对应的标签文件。")

    if labels_missing_images:
        print(f"\n共有 {len(labels_missing_images)} 个标签文件缺少对应的图片：")
        for fname in sorted(labels_missing_images):
            print(f"  - {fname}")
    else:
        print("所有标签文件都有对应的图片。")

    # 可选：将结果保存到文件
    with open("correspondence_check_result.txt", "w") as f:
        if images_missing_labels:
            f.write(f"共有 {len(images_missing_labels)} 张图片缺少对应的标签文件：\n")
            for fname in sorted(images_missing_labels):
                f.write(f"  - {fname}\n")
        else:
            f.write("所有图片都有对应的标签文件。\n")

        if labels_missing_images:
            f.write(f"\n共有 {len(labels_missing_images)} 个标签文件缺少对应的图片：\n")
            for fname in sorted(labels_missing_images):
                f.write(f"  - {fname}\n")
        else:
            f.write("所有标签文件都有对应的图片。\n")

    print("\n对应关系检查结果已保存到 'correspondence_check_result.txt'。")

if __name__ == "__main__":
    main()
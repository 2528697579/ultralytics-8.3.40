from ultralytics import YOLO

# 初始化模型（例如，使用预训练的 YOLOv8n 模型）
model = YOLO("yolo11s.pt")  # 根据需求选择合适的模型

# 训练模型，确保参数名称正确
model.train(data="ONE_DATA.yaml",
            # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
            cache=False,
            imgsz=640,
            epochs=300,
            single_cls=False,  # 是否是单类别检测
            batch=14,
            close_mosaic=0,
            workers=0,
            device='0',
            optimizer='AdamW',  # using SGD 优化器 默认为auto建议大家使用固定的.
            # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
            amp=True,  # 如果出现训练损失为Nan可以关闭amp
            project='runs/train',
            name='one_AdamW_train',
            patience=20  # 早停的耐心值
            )

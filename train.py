from ultralytics import YOLO

def main():
    # 1. 加载预训练模型 (我们这里使用你已经下载好的 yolo26n.pt)
    # 对于边缘部署来说，n (nano) 版本在速度和精度的平衡上往往是最佳选择
    model = YOLO('yolo26n.pt')

    # 2. 开始训练
    results = model.train(
        data='./yolo26_dataset/data.yaml', # 指向刚才修改好的配置文件
        epochs=200,                        # 训练轮数
        imgsz=640,                         # 图像输入尺寸，这也是将来转 Hailo 模型的标准尺寸
        batch=16,                          # 批次大小，如果你的显存不够（比如报 OOM 错误），可以改成 8 或 4
        device=0,                          # 0 表示使用第一块 NVIDIA 独立显卡进行训练
        project='runs/train',              # 训练结果保存的主目录
        name='yolo26_custom_model',        # 本次训练结果保存的子文件夹名
        workers=4,                         # 数据加载的线程数，Windows 下如果报错可以尝试设为 0
        patience=20                        # 早停机制：如果 20 轮验证集精度没有提升，则自动停止
    )

if __name__ == '__main__':
    main()
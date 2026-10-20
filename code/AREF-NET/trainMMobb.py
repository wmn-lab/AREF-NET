#这是提供 参考示例的 YOLOMM 多模态旋转框训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM

if __name__ == '__main__':
    # 可根据需要替换为你自己的多模态 OBB 配置/权重
    model = YOLOMM('yolo11n-mm-obb.yaml')
    model.train(
        data='/home/zhizi/work/multimodel/ultralyticmm/datasets/obb/obb.yaml',
        epochs=20,
        batch=16,
        # modality='X',  # 模态消融参数 非必要不得开启
        # cache=True,
        exist_ok=True,
        project='ResTest',
        name='Test/OBB_MM'
    )

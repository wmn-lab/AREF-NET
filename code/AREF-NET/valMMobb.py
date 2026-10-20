# 这是提供 参考示例的 YOLOMM OBB 验证器使用方法
# 按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM

if __name__ == "__main__":
    model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/Test/OBB_MM/weights/best.pt')
    model.val(
        data='/home/zhizi/work/multimodel/ultralyticmm/datasets/obb/obb.yaml',
        split='val',
        device='0',
        # modality='x',  # 模态消融参数，非必要不要开启
        project='ResTest',
        name='Val_OBB_MM',
    )

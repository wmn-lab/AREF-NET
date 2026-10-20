#这是提供 参考示例的 YOLOMM验证器使用方法
#按必须按照你的需求来更改而不是盲目去使用
from ultralytics import YOLOMM

model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/YOLOMM12/weights/best.pt')
model.val(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
          split='test',device='1',
        #   modality='x',模态消融参数 非必要不得开启
          project='ResTest',
          name='Val')
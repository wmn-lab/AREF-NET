from ultralytics import RTDETRMM

model = RTDETRMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/RTDETR-mid/weights/best.pt')
model.cocoval(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
          split='val',device='0',half=True,
        #   modality='x',
          project='ResTest',
          name='Val')
import os
from ultralytics import YOLOMM

# CUDA 设备控制
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/test/YOLOMM_seg13/weights/best.pt')
data = '/home/zhizi/work/multimodel/ultralyticmm/data_SEG.yaml'
model.val(data=data,
          split='test',device='0',
        #   modality='x',
          project='ResTest',
          name='Val')
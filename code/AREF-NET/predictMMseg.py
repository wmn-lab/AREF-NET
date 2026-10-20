import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLOMM

# 使用可用的YOLOMM权重
model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/test/YOLOMM_seg13/weights/best.pt')
rgb_img = '/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png'
x_img = '/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png'
# 测试双模态预测
print("=== 测试双模态预测 ===")
model.predict([rgb_img,
               x_img
               ],
              project='ResTest',
              name='yolomm_dual_modal',
              save=True,
              exist_ok=True
              )

# 测试单模态RGB预测
print("\n=== 测试单模态RGB预测 ===")
model.predict(rgb_img,
              project='ResTest',
              name='yolomm_single_rgb',
              save=True,
              modality='rgb',
              exist_ok=True
              )

# 测试单模态X模态预测  
print("\n=== 测试单模态X模态预测 ===")
model.predict(x_img,
              project='ResTest',
              name='yolomm_single_x',
              save=True,
              modality='x',
              exist_ok=True
              )
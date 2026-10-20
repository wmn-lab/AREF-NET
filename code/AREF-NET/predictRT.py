import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import RTDETRMM

# 使用可用的RT-DETR权重
model = RTDETRMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/RTDETR-early/weights/best.pt')

# 测试双模态预测
print("=== 测试双模态预测 ===")
model.predict(['/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',
               '/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png'
               ],
              project='ResTest',
              name='rtdetrmm_dual_modal',
              save=True,
              exist_ok=True
              )

# 测试单模态RGB预测
print("\n=== 测试单模态RGB预测 ===")
model.predict('/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',
              project='ResTest',
              name='rtdetrmm_single_rgb',
              save=True,
              modality='rgb',
              exist_ok=True
              )

# 测试单模态X模态预测  
print("\n=== 测试单模态X模态预测 ===")
model.predict('/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png',
              project='ResTest',
              name='rtdetrmm_single_x',
              save=True,
              modality='x',
              exist_ok=True
              )
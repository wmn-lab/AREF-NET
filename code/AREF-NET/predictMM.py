# #这是提供 参考示例的 YOLOMM推理器使用方法
# #按必须按照你的需求来更改而不是盲目去使用


# from ultralytics import YOLOMM
# model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/YOLOMM/weights/best.pt')

# # 测试双模态预测
# print("=== 测试双模态预测 ===")
# model.predict(['/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',
#                '/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png'
#                ],
#               project ='ResTest',
#               name='dual_modal_test',
#               save=True,
#               exist_ok=True
#               )

# # 测试单模态RGB预测
# print("\n=== 测试单模态RGB预测 ===")
# model.predict('/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',
#               project ='ResTest',
#               name='single_modal_rgb_test',
#               save=True,
#               modality='rgb',
#               exist_ok=True
#               )

# # 测试单模态红外预测  
# print("\n=== 测试单模态红外预测 ===")
# model.predict('/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png',
#               project ='ResTest',
#               name='single_modal_ir_test',
#               save=True,
#               modality='X',
#               exist_ok=True
#               )


from ultralytics import YOLOMM
model = YOLOMM('/home/wmn/code/mm/MutilModel9.4/zheda_paper/train/mid/weights/best.pt')

# 测试双模态预测
print("=== 测试双模态预测 ===")
model.predict(['/home/wmn/code/data/zheda_water/images/val/3910.png',
               '/home/wmn/code/data/zheda_water/images_edge/val/3910.png'
               ],
              project ='zheda_paper',
              name='predict',
              save=True,
              exist_ok=True
              )




# 测试单模态RGB预测
# print("\n=== 测试单模态RGB预测 ===")
# model.predict('/home/wmn/code/data/zheda_water/images/val/89.png',
#               project ='zheda_paper',
#               name='predict',
#               save=True,
#               modality='rgb',
#               exist_ok=True
#               )

# # 测试单模态红外预测  
# print("\n=== 测试单模态红外预测 ===")
# model.predict('/home/wmn/code/data/zheda_water/images_edge/val/89.png',
#               project ='zheda_paper',
#               name='predict',
#               save=True,
#               modality='edge',
#               exist_ok=True
#               )
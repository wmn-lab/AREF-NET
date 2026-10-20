from ultralytics import RTDETRMM

model = RTDETRMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/RTDETR8/weights/best.pt')
model.val(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
            split='test',device=1,
            # modality='rgb',模态消融参数 非必要不得开启
            project='ResTest',name='RTDETRval')
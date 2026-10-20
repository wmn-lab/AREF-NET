from ultralytics import RTDETRMM

model = RTDETRMM('rtdetr-r18-mm-mid.yaml')
model.train(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
            epochs=50,device=1,batch=16,
            # modality='rgb', 模态消融参数 非必要不得开启
            # cache=True,
            project='ResTest',name='RTDETR-mid')
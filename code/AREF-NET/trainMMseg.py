#这是提供 参考示例的 YOLOMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM

if __name__ == '__main__':
    
    model = YOLOMM('yolo11n-mm-mid-seg.yaml')
    model.train(data='/home/zhizi/work/multimodel/ultralyticmm/data_SEG.yaml',
                epochs=3,batch=8*2,
                # modality='X', #模态消融参数，非必要不用开启
                cache=True,
                # amp = False,
                # exist_ok=True,
                project='ResTest',name='test/YOLOMM_seg')

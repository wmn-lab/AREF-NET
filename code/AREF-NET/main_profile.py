import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLOMM

if __name__ == '__main__':
    # choose your yaml file
    model = YOLOMM('yolo11n-mm-mid.yaml')
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    model.fuse()
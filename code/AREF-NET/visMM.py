#这是热力图可视化的方法，请你参考我的使用
#使用方法查看 wiki目录中的vis_usage.md文件
from ultralytics import YOLOMM

model = YOLOMM('/home/wmn/code/mm/MutilModel9.4/zheda_paper/train/mid/weights/best.pt')
model.vis(
          rgb_source = '/home/wmn/code/data/zheda_water/paper/rgb/358.jpg',  # rgb_source：RGB 输入
          x_source = '/home/wmn/code/data/zheda_water/paper/rgb/358.jpg',  # x_source：X 模态输入
          method='feature',  #热力图
          layers=[29,32,35,38],  # 使用yaml层
          overlay='rgb',  # 叠加底图：'rgb'|'x'|'dual'，默认'rgb'；仅传X时自动改为'x'
          # modality='X',  # 模态消融：'rgb' 或 'x'；双模态输入时也可强制消融
          alg='gradcam',
        #   split=True,
          save=True,
          project='ResTest/vis',
          name='Vis')

# 这是提供 参考示例的 YOLOMM OBB 推理器使用方法
# 按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM


if __name__ == "__main__":
    model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/Test/OBB_MM/weights/best.pt')

    # 示例路径，请替换为你的实际 RGB / X 模态测试图片
    # 默认示例路径（数据集中已存在的配对样本）
    rgb_img = '/home/zhizi/work/multimodel/ultralyticmm/datasets/obb/images/train/1.png'
    x_img = '/home/zhizi/work/multimodel/ultralyticmm/datasets/obb/images_sar/train/1.png'

    # 推理阈值可根据需要调整（低阈值便于观察检出情况）
    common_kwargs = dict(project='ResTest', save=True, exist_ok=True, conf=0.01, iou=0.35)

    print("=== 测试双模态 OBB 预测 ===")
    model.predict(
        [rgb_img, x_img],
        name='OBB_MM_dual_modal',
        batch=2,  # 确保两张图打包成同一 batch，触发双模态预处理
        **common_kwargs,
    )

    print("\n=== 测试单模态 RGB OBB 预测 ===")
    model.predict(
        rgb_img,
        name='OBB_MM_single_rgb',
        modality='rgb',
        **common_kwargs,
    )

    print("\n=== 测试单模态 X OBB 预测 ===")
    model.predict(
        x_img,
        name='OBB_MM_single_x',
        modality='x',
        **common_kwargs,
    )

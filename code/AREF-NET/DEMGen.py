"""
DEMGen 离线 DEM 模态生成示例脚本。

用法与 depthGen.py 对齐：只需初始化 DEMGen，然后 run(data.yaml)。

默认保存目录：
- 当 save_dir=None 且输入为 data.yaml 时，输出到 data.yaml 对应的 images_dem/<split>/ 下。

提示：
- DEM 特征为 6 通道，训练/推理时请在 data.yaml 中设置：
  - modality: {rgb: images, dem: images_dem}
  - x_modality: dem（可选）
  - Xch: 6
"""

from ultralytics import DEMGen


if __name__ == "__main__":
    gen = DEMGen(
        source_modality="rgb",  # 从哪个模态生成 DEM 特征，默认 rgb
        device="cuda:0",
        save_dir=None,          # None: 默认保存到 images_dem/<split>
        split=None,             # 可选: "train" / "val" / "test" / "train,val"
        save_format="npy",      # 'npy' | 'npz' | 'png'
        batch_size=1,
        num_workers=0,
        # 生成参数（与 tools/generate_dem_features.py 对齐）
        gaussian_ksize=3,
        sobel_ksize=3,
        roughness_ksize=5,
        local_diff_ksize=15,
        use_scharr=False,
    )

    stats = gen.run("/home/zhizi/work/multimodel/ultralyticmm/datasets/tree/data.yaml")
    print(f"生成完成: 成功 {stats.success}/{stats.total}, 失败 {stats.failed}")
    if stats.failures:
        print("失败列表:")
        for path, err in stats.failures:
            print(f"- {path}: {err}")

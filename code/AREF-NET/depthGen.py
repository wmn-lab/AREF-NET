from ultralytics import DepthGen

if __name__ == "__main__":

    gen = DepthGen(
        weights='ref/Depth-Anything-V2-main/checkpoints/depth_anything_v2_vitb.pth', #填权重地址
        device='cuda:0',
        save_dir=None,  # None: 默认保存到 data.yaml 对应 images_depth/<split>
        save_comparison=True,  # 对比图存到 images_depth_compare/<split>
        save_npy=False,
        save_color=True,
        save_raw16=False,
        split=None,  # 可选: "train", "val", "test" 或 "train,val"
        batch_size=1,
        num_workers=0,
    )

    stats = gen.run('/home/zhizi/work/multimodel/ultralyticmm/datasets/rgbt_tiny_yolo/dataset.yaml')
    print(f"生成完成: 成功 {stats.success}/{stats.total}, 失败 {stats.failed}")
    if stats.failures:
        print("失败列表:")
        for path, err in stats.failures:
            print(f"- {path}: {err}")

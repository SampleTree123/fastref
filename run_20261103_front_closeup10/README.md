# 10个正面近景的VAE还原检查

从合成测试集的96个唯一面部近景视频中，随机不放回抽10个，seed=42。人工检查发现3条主要为侧脸/背面，已排除并以seed=43继续随机补抽3条正面案例；最终清单及排除编号见cases.json。按视频去重后抽样，不按编辑样本重复抽同一个视频。镜头为camera_id=10（正面面部近景，跟随头部位置）。

2026-11-03 19:46 SGT：最终10例全部完成，先四卡并行，随后单卡补做3例。30个视频与60张PNG均核对通过，共226.9MiB。每条视频使用当前训练预处理：12fps、121帧，横屏768×448、竖屏448×768。复用官方VAE编码器和目前推理使用的官方VAE解码器，BF16；不加载生成模型或LoRA，不去噪，不更新权重。

页面每个案例三列：原始视频、训练尺寸的VAE输入、VAE还原。左右的PNG对照提供第0、5、10秒的无损截图，可用来排除MP4再次压缩的影响。两份处理后视频使用同一H.264/crf16设置。第10秒沿用训练采样的末帧保持。

`cases.json`记录抽样来源；`state.json`保存输入尺寸、采样帧号、VAE形状、误差与检查结论；已完成的临时日志和排除案例输出已清理。整个`display/`可下载，里面保存实际视频与PNG。像素PSNR不等于人脸识别正确率，最终清晰度由页面人工检查。

## 调用方式

```bash
cd /scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
bash scripts/common/run_trainer_python.sh scripts/inference/run_vae_reconstruction.py \
  --cases experiments/inference/vae_reconstruction/run_20261103_front_closeup10/cases.json \
  --output experiments/inference/vae_reconstruction/run_20261103_front_closeup10 \
  --workers 4
```

本次已经启动，请勿重复运行同一目录。重复实验需要新的输出目录。调用顺序：`run_vae_reconstruction.py` → `prepare_datamake_media.py`抽帧、缩放与补边 → 官方`load_video_vae_encoder` → 官方`VideoDecoder` → 官方`encode_video`写MP4、PIL保存PNG → 生成本地展示页面。

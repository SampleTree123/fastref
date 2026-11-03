# 两组第150步检查点的实际推理

使用189814中的一张H200，依次生成原部件多图、套装正背分开两组结果，各16例。两组都为第150步，不与第一次682步训练混称同等训练程度。

参数：40步、seed42、CFG3、STG1、rescale0.7、已有负面提示词。检查点调用前固定保留；运行不修改训练参数。

- `manifest.json`：两组配置、检查点和输出位置。
- `state.json`、`logs/`：执行状态和日志。
- `original/`、`outfit/`：生成视频和记录。
- `gallery/index.html`：两组同页对比；未完成组显示待生成。
- `display/index.html`：全部完成后导出的完整页面，媒体为实际文件，可整目录下载。

调用：在项目根目录执行 `python experiments/inference/checkpoint_comparison/run_20261103_step150/run.py`，需要已分配的空闲GPU。本轮已在189814启动时不要重复运行。

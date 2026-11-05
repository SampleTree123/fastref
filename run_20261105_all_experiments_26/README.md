# 八组训练与视觉参考强度对比

2026-11-05 20:50 SGT：全部完成，104条新结果＋104条复用结果，完整[display/index.html](display/index.html)已导出。媒体存在检查通过，整目录可下载。

2026-11-05。使用193061的8张H200，复用11月4日同一批26例（7类UE编辑各3条＋CoinVE5条），合并所有本批训练方案。每例显示源视频、真值和八组输出，每行最多四组；窄屏自动换行。

| 组 | 检查点 | 参考CFG | 来源 |
|---|---|---:|---|
| 首轮原多图＋补边＋原时间 | 682 | 1 | 复用昨天结果 |
| 原部件多图＋连续时间 | 512 | 1 | 复用昨天结果 |
| 套装正背＋连续时间 | 512 | 1 | 复用昨天结果 |
| 拼图＋连续时间 | 768 | 1 | 复用昨天结果 |
| 套装正背＋原时间（191872） | 768 | 1 | 本次生成 |
| 套装正背＋连续时间＋参考丢弃训练（191873） | 768 | 1 | 本次生成 |
| 同一个191873模型 | 768 | 3 | 本次生成 |
| 镜头首帧＋套装正背＋连续时间（191904） | 启动时最新完整检查点，见manifest | 1 | 本次生成 |

本次新增104条，合并后208条。参考CFG1保留全部输入参考，不额外放大视觉引导；参考CFG3增加移除额外参考的一次预测来放大参考作用，源视频保留，不随机丢弃输入。没有额外参考的样本，两者应走相同生成路径。

共同推理参数：40步Euler，seed42，文本CFG3，STG1/block28，guidance rescale0.7，guidance batch1，负面提示词同昨天。UE121帧、CoinVE57帧、12fps；横屏768×448、竖屏448×768。camera组的镜头样本按它自己的数据增加首帧参考和相应prompt；各组训练步数不同，不能把跨组差异全部解释成单一因素。

## 文件

- `manifest.json`：八组的检查点、数据、配置和指导参数。
- `cases.json`、`inference.json`：从昨天逐字复制的案例与公共参数。
- `*_config.yaml`：本次推理用的对应训练配置副本。
- `run.py`：复用四组旧结果，在八卡生成四组新结果，更新网页并导出。
- `launch.sh`：在已经分配的Slurm任务内启动run.py。
- `state.json`：当前阶段与完成状态；`logs/`和各组`gpu_*.log`是运行日志。
- `gallery/index.html`：运行中的页面，媒体可能引用原结果目录。
- `display/index.html`：完成后的完整展示。下载整个display文件夹即可，包含实际视频和参考图，没有依赖外部目录的媒体链接。

调用路径：`launch.sh JOB_ID 8` → srun → 原生环境 → run.py → run_fixed_cases_parallel.py → 每卡run_fixed_case_inference.py → 原生模型/调度器 → build_experiment_comparison.py → export_guidance_gallery.py。

```bash
cd /scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
bash experiments/inference/checkpoint_comparison/run_20261105_all_experiments_26/launch.sh 193061 8
```

本批已启动，不要重复执行命令。完成后不会自动释放193061，也不会自动续训191986。191986已按用户要求停止，50步恢复文件仍保留。

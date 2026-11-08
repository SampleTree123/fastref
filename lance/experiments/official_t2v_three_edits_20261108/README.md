# Lance官方基座：文生视频后连续编辑三次

2026-11-08。测试已完成：1段T2V＋3次连续V2V，共4段848×480、61帧、12fps视频。用户授权的独立小测试，不恢复旧LoRA训练。使用195783的一张空闲H200，不停止或释放整个交互作业。

流程：文字生成白上衣黑发女性在花园行走 → 上衣改为蓝色 → 头发改为金色 → 背景改为城市。每轮只给上一轮实际视频和当前文字，使用官方单轮video_edit入口。

官方代码固定为4baeee0，单独放在runtime/official_4baeee0，未应用DataMake补丁；downloads只是指向官方权重的软链接。权重版本在model_revision.json。复用原xks_lance环境，不加载旧训练LoRA；外部文字增强关闭。

## 实际结果

[打开展示](index.html)。三次编辑的输入SHA均与上一轮输出一致，四个视频均逐帧解码通过。观察四段视频的2秒帧，改蓝上衣与改金发可见；最后一段另查看0.5秒和4秒，仍保留花园，没有明显换成城市街道。不是总体成功率评价，未为了得到成功案例而重抽seed或更换prompt。三次编辑各约181秒，包含模型重新加载；T2V官方生成批次约65.92秒，不包含加载。

首次编排在T2V成功后因环境没有av库而停止；已改用已有OpenCV检查视频，显式恢复并复用该T2V，没有重新生成。runner_attempt1.log保留原因；后续步骤正常exit0。没有改动官方已跟踪源码，也没有加载LoRA。

单卡推理步骤已结束；整个195783交互资源未关闭。

## 文件与使用

- download_weights.py：只下载官方视频模型、ViT和VAE；download.log记录下载，model_revision.json记录版本与大小检查。
- run.sh：激活原环境后调用run_chain.py。
- run_chain.py：依次调用官方inference_lance.sh → inference_lance.py → 官方ValidationDataset与模型生成；它只编排文件与生成次序，不修改网络。
- round_00到round_03：每次的输入、实际命令、官方日志和生成视频。
- state.json：完成状态、文件SHA、真实前后依赖、分辨率、帧数和时间。
- index.html：随完成进度更新的展示；下载本目录及round_*的generated视频即可离线查看。

在现有授权资源上使用：

```bash
cd /scratch/prj0000000275/jcy/xks/DataMake
srun --jobid=195783 --overlap --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  bash 10_research/model_experiments/lance/experiments/official_t2v_three_edits_20261108/run.sh
```

脚本默认拒绝覆盖已存在state.json；确认中断原因后可使用--resume复用完整输出。不要重复提交正在运行的测试。参数为480p、请求61帧、30步、文字CFG4、shift3.5、seed42。实际帧数和尺寸以state.json为准，官方V2V会按输入视频处理。

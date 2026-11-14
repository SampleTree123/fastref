# H3与VACE加速小样

## 2026-11-14 16:32 SGT：两个模型均完成完整视频生成

同一换身份例子，源视频＋一张人物参考。两份LoRA均已实际加载并完成生成，所有输出帧均通过解码检查；202767的测试进程已退出，四卡显存均归零。未启动正式100链。

| 测试 | GPU | 生成耗时（不含模型加载） | 模型加载 | 输出 |
|---|---|---|---|---|
| VACE＋LightX2V LoRA，4步 | 1×H200 | 224.1秒，约3分44秒 | 19.2秒 | 121帧，1280×736，12fps |
| H3 Ref2VA＋PAI LoRA，8步 | 4×H200 | 请求155.7秒，约2分36秒；服务端151.3秒 | 78.3秒 | 243帧，1344×768，24fps |

[查看页面](index.html)同时包含源视频、人物参考图和两种结果，下载整个本目录即可离线查看。只想查看视频可直接打开[vace_4step/output.mp4](vace_4step/output.mp4)和[h3_8step/output.mp4](h3_8step/output.mp4)。

**能运行不等于效果合格。** 抽查首、中、末帧：VACE有正常视频画面；H3明显改变了本应保留的服装、发型，出现额外身体暴露，不能据此直接用于正式评估。未判断问题究竟来自基模、加速权重还是提示词，后续需要与原50步同案例结果对照。当前只完成可运行性验收。

同案例旧记录：VACE40步3827.8秒（约63.8分钟），H3原50步841.4秒（约14.0分钟）。本次分别约快17.1倍与5.4倍；是历史运行时间对照，硬件并行和计时范围不同，且不是质量等价结论。原记录快照在[baseline_timing.json](baseline_timing.json)。

VACE原1457个张量中，1215个由官方Diffusers/PEFT加载并逐项核对；另外242个小向量由明确映射补入。H3258/258个权重已匹配，日志确认8个配套输出头加载。其SGLang为官方仓库commit `fa179312e94c764f8637724ea2a364d302888906` 的现有部署；转换工具无本地改动。

H3首次启动遇到HTTP端口＋1冲突，未生成视频；已改为检查低位连续端口对，重试成功。新功能通过可选pdd_dir接入h3_service.py，原50步配置不变。当前正式VACE40步入口仍未切换加速版本。

## 测试设置与实现

目标：确认用户选定的两份加速权重能用于已有参考编辑流程，生成完整可播放的视频。只验证一个换身份例子`identity_01`，不据此认定所有任务效果合格，也不启动100条链。

使用202767的空闲H200，VACE单卡，H3四卡顺序执行。借用师兄现有基模、环境和已准备条件，未更改共享模型或环境。输入只有源视频、编辑文字和人物参考图；不把目标视频作为条件。

| 项目 | VACE | H3 |
|---|---|---|
| 基模 | Wan2.1-VACE-14B | MiniMax-H3 Base Ref2VA |
| 加速权重 | Kijai发布的Wan21 T2V14B LightX2V rank32 | Alibaba PAI Ref2VA Acc8Step |
| 步数 | 4 | 8 |
| 其他设置 | CFG1，LoRA1，shift5，seed42 | video shift12，audio shift3，seed42 |
| 输出 | 1280×736，121帧，12fps | 1344×768，243帧，24fps |
| 输入准备 | 沿用既有皮肤mask，编辑区填128，保留背景和参考图 | 已准备的源视频和参考图 |

精确下载版本和SHA256见[downloads.json](downloads.json)。H3是PDD权重，除LoRA之外还需要专用输出层。已直接使用现有SGLang的构建和输出层合并工具：258/258个权重匹配，32个输出头按配套规则合成8步；日志为`h3_merge.log`、`h3_fuse.log`。不是把普通50步配置的数字随意改成8。

VACE的额外参数不能忽略：当前官方加载器会跳过242个小向量，本地适配器补齐它们，再使用官方LoRA加载方式。成功运行后，`vace_4step/lora_load.json`记录全部加载核对结果。

结果和耗时分别写入`vace_4step/result.json`、`h3_8step/result.json`；视频均叫`output.mp4`。脚本入口和调用顺序见使用说明位于原 DataMake 仓库：`10_research/model_experiments/qwen3.8/scripts/acceleration/README.md`。模型加载时间与生成时间分开记录；本次不测Qwen、动态分割和多人并行的全流程耗时。

资料（2026-11-14核对）：[LightX2V模型说明](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)、[四步配置](https://github.com/ModelTC/LightX2V/blob/main/configs/distill/wan21/wan_t2v_distill_model_4step_cfg.json)、[调度实现](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/schedulers/wan/step_distill/scheduler.py)、[H3加速权重](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)。

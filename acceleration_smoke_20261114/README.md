# H3与VACE加速小样

## 2026-11-14 16:57 SGT：服装与发色四个新增视频全部完成

[统一查看页面](index.html)保留原身份例子，并新增整套服装whole_outfit_01和粉色发色hair_color_03；每组并排显示源视频、VACE和H3，服装另附正背两张参考图。整个本目录可下载后直接查看，媒体链接使用相对路径。

|新增例子|VACE4步，单卡H200|H3 Ref2VA8步，四卡H200|
|---|---|---|
|整套服装|235.6秒，约3分56秒|160.5秒，约2分41秒|
|发色改粉色|213.2秒，约3分33秒|145.4秒，约2分25秒|

耗时不含模型加载和预先制作mask。VACE两例在两张卡并行；H3两例共用一个服务依次生成，模型加载共72.3秒。VACE输出各121帧，H3各243帧，均通过完整解码检查。202767模型进程已结束，四卡显存归零；交互资源仍保留。

服装抽查首、中、末帧：两者均改为参考中的粉色套装；VACE袖子等细节不同于参考，H3背景取景也有变化。发色抽查：VACE头发变粉色、保留了总体构图；H3虽改成粉色，但输出出现三分屏，未保持原视频构图。仅作初步观察，完整效果以页面视频为准。数据、seed42和加速参数沿用上一轮。

结果回执见各案例目录中的`vace_4step/result.json`和`h3_8step/result.json`，汇总为[all_cases_results.json](all_cases_results.json)。额外脚本修改的版本记录在[additional_code_hashes.json](additional_code_hashes.json)。

## 2026-11-14 16:45 SGT：新增服装与发色，正在执行

用户要求再测两个任务。选用whole_outfit_01（整套服装，正背两张图）及hair_color_03（改成粉色，无参考图）；各用VACE4步与H3 Ref2VA8步，seed42及其余设置沿用。先两张空闲卡并行VACE，再四卡H3共用一次模型加载。原身份结果保留，最后合并在同一个页面。

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

结果和耗时分别写入`vace_4step/result.json`、`h3_8step/result.json`；视频均叫`output.mp4`。脚本入口和调用顺序见[使用说明](../../scripts/acceleration/README.md)。模型加载时间与生成时间分开记录；本次不测Qwen、动态分割和多人并行的全流程耗时。

资料（2026-11-14核对）：[LightX2V模型说明](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)、[四步配置](https://github.com/ModelTC/LightX2V/blob/main/configs/distill/wan21/wan_t2v_distill_model_4step_cfg.json)、[调度实现](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/schedulers/wan/step_distill/scheduler.py)、[H3加速权重](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)。

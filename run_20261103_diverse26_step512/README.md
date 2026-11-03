# 新一批26个推理案例

这批使用两组最终512步检查点。21个合成案例按七种编辑各选三个，另选五个真实案例。旧案例保留在原来的页面，本页全部换新。

合成案例来自测试集；同一种编辑尽量更换人物、场景和穿搭，并兼顾动作、镜头与横竖屏。这里用于看效果，不据此计算总体评测分数。

|编辑|案例|人物编号|场景|穿搭系列|动作|镜头|画幅|
|---|---|---|---|---|---|---|---|
|人物身份|29829|17|未来都市|Mesh it Up|举拳庆祝|正面推进|portrait|
|人物身份|30872|12|海滨小镇|Pretty Dresses|扶额|面部近景（柔和跟随头部位置）|landscape|
|人物身份|30051|19|傍晚公路|1920S Collection|向前行走|正面推进|portrait|
|发型|30247|15|维多利亚街道|Off Shoulder Dresses|打喷嚏|后侧面上半身|portrait|
|发型|27720|13|日本城市街区|Streetwear Collection|摆姿|仰视全身|landscape|
|发型|29004|18|游乐园|Mesh it Up|思考|背面拉远|landscape|
|发色|30522|20|日本城市街区|1920S Collection|打喷嚏|360°全身环绕|portrait|
|发色|26754|19|未来都市|Summer|摆姿|侧面推进|landscape|
|发色|28170|14|游乐园|Shirts Pants A I|扶额|俯视全身|landscape|
|服装|28215|18|傍晚公路|1920S Collection|举拳庆祝|前侧全身|portrait|
|服装|29188|12|海滨小镇|Pretty Dresses|向前行走|前侧上半身|portrait|
|服装|27118|17|未来都市|Summer|思考|背面全身|landscape|
|动作|27599|13|傍晚公路|Pretty Dresses|举拳庆祝|背面上半身|landscape|
|动作|27897|20|海滨小镇|Summer|向前行走|120°全身环绕|portrait|
|动作|26652|19|维多利亚街道|Summer|摆姿|正面拉远|landscape|
|镜头|28537|13|未来都市|Pretty Dresses|扶额|正面全身|portrait|
|镜头|31550|20|傍晚公路|1920S Collection|思考|侧面上半身|portrait|
|镜头|29259|19|日本城市街区|Mesh it Up|打喷嚏|侧面全身|landscape|
|场景|30129|15|维多利亚街道|Streetwear Collection|摆姿|后侧面全身|portrait|
|场景|29043|18|游乐园|Jiving|举拳庆祝|侧面拉远|landscape|
|场景|29758|14|未来都市|Off Shoulder Dresses|摆姿|前侧全身|landscape|

真实案例换成跑车、沿海道路、运动员、厨房近景和玩偶五种内容，避免全是相似的人物场景。

- coinve:151368：跑车：轮毂与山路背景。
- coinve:197535：沿海道路：树木与天空。
- coinve:83632：运动员：球衣颜色与背景。
- coinve:57742：厨房近景：盆栽与窗户。
- coinve:193307：玩偶：衣服、装饰与背景风格。

## 当前状态

2026-11-03 18:44 SGT：两组各26例全部完成，共52条生成视频，总耗时约33分钟。完整展示已导出到`display/`，133个媒体文件全部存在且不是软链接，可以整体下载后打开`index.html`。模型效果等待人工查看。

保持40步、CFG 3、STG 1、rescale 0.7、seed 42和原负面提示词。

`cases.json`是固定案例清单；`selection.json`记录属性和旧案例排除情况；`manifest.json`指向两组检查点。
`original_config.yaml`与`outfit_config.yaml`沿用对应训练推理配置，`inference.json`保存共同推理参数。

页面的 `gallery/index.html` 可先查看输入；生成完成后再导出包含所有实际视频的 `display/`，供整文件夹下载。

## 执行入口（本次已完成，请勿重复执行）

本次由190301内的后台srun执行。下面是内部入口，保留方便查看；正在运行时不要重复执行。

```bash
cd /scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
bash scripts/common/run_trainer_python.sh experiments/inference/checkpoint_comparison/run_20261103_diverse26_step512/run.py
```

`run.py`先保留两份512步检查点，然后依次调用`run_fixed_cases_parallel.py`，后者调用单卡`run_fixed_case_inference.py`。读取`manifest.json`中的数据清单与配置，用`native_cached_dataset.py`读编码、`native_reference_inference.py`调用官方模型。最后`build_experiment_comparison.py`生成对比页，`export_guidance_gallery.py`复制完整展示文件夹。本次每组使用四张卡分配案例，两个模型依次执行。状态见`state.json`，各卡日志见`original/gpu_*.log`或`outfit/gpu_*.log`。

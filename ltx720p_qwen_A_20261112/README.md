# 720p 三模型与 Qwen 多轮编辑测试

先打开 `index.html` 比较固定 27 个单轮案例。A 与 B 各显示参考 CFG 1、3；C 只显示参考 CFG 1。文本 CFG 都是 3，STG 为 1，生成 40 步，初始 seed 为 42。没有额外参考的案例在 CFG 3 列复用同模型 CFG 1 的视频。

单轮结果的原始记录在 `A_ref1/`、`A_ref3/`、`B_ref1/`、`B_ref3/`、`C_ref1/`；`gallery.json` 记录网页对应的来源。7 个案例属于新混合训练集，只供展示，不计入独立测试结果。

多轮结果见 `qwen_A_retry3/report/index.html` 和 `qwen_A_retry3/report/README.md`。这部分固定选择 12 条旧测试链、共 39 轮，用 A 模型。每轮检查编辑结果；若未通过，seed 改为 43、44、45 重试。四次仍未通过时停止当前链，其他链继续。每次尝试保存在对应 `cases/链编号/turn_XX/progress.json`。

`campaign.json` 显示实际运行进度，`smoke/verified.json` 是四卡短测通过的标记。生成和自动检查通过不等于人工质量验收。

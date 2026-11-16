# 明亮场景竖屏展示8例

2026-11-16：用户允许使用训练节点。这批用于挑选展示效果，不纳入正式测试成绩。
人物按数据集WF/WM标签筛选，4位女性案例和4位男性案例。场景经过抽帧人工查看，选白天取景。
各链均为：原视频0改摆姿势→视频1改粉发→回原视频0换服装。目标服装的全部部件编号均未出现在完整测试清单中。
服装参考和视频可能参与过训练，并非未见素材测试；原有正式100链不变。

| 候选 | 原节点 | 人物标签 | 场景 | 镜头 | 原发色 | 原动作 | 换入服装 |
|---|---:|---|---|---|---|---|---|
| showcase_bright_portrait_01 | 3238 | CFD40_WF_005 | 京都小巷 | 正面拉远 | 黑色 | 喘气恢复 | 青蓝色上衣与短裤 |
| showcase_bright_portrait_02 | 3251 | CFD40_WF_005 | 京都小巷 | 正面拉远 | 黑色 | 喘气恢复 | 黑白上衣与白色长裤 |
| showcase_bright_portrait_03 | 4649 | CFD40_WF_004 | 中世纪城堡 | 正面拉远 | 白色 | 打招呼B | 金色上衣与蓝色长裤 |
| showcase_bright_portrait_04 | 4663 | CFD40_WF_004 | 中世纪城堡 | 正面拉远 | 白色 | 欢呼跳跃 | 彩色上衣与印花长裤 |
| showcase_bright_portrait_05 | 3828 | CFD40_WM_005 | 维多利亚街道 | 正面推进 | 蓝色 | 打招呼A | 橙色衬衫与绿色短裤 |
| showcase_bright_portrait_06 | 3850 | CFD40_WM_004 | 维多利亚街道 | 正面推进 | 蓝色 | 打招呼A | 深色正装套装 |
| showcase_bright_portrait_07 | 5626 | CFD40_WM_002 | 日本城市街区 | 前侧全身 | 黑色 | 随乐舞动 | 白蓝色休闲套装 |
| showcase_bright_portrait_08 | 5645 | CFD40_WM_002 | 红色沙漠 | 前侧全身 | 黑色 | 鞠躬 | 蓝色格纹衬衫与浅色长裤 |

服装指令直接复用该套装的正式文本，包含正背面说明和赤足/鞋子规则。第二轮没有另加参考；第三轮应恢复原发色和原动作。
case_cards.json列出七属性、参考编号和有无真值；没有匹配节点的轮次不提供真值。verification.json保存素材校验值。
构造入口：scripts/memory_comparison/prepare_bright_portrait8.py；读取正式nodes、conditions、catalogs及train/test清单。

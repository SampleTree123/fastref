# 正式理解测试的 16 个编号不匹配案例

## 2026-11-08T03:45:50+08:00：结果与范围

1000 题全部运行完成。按冻结答案编号评分，正确 984 题，错误 16 题，准确率 **98.4%**。16 题都来自“按画面描述找历史版本”；这类题共 500 个，正确 484 个。其余 500 个直接编号、前后关系或普通续编题全部对齐答案。

| 找历史时使用的属性 | 题数 | 正确 | 错误 |
|---|---:|---:|---:|
| 镜头 | 250 | 239 | 11 |
| 发色 | 250 | 245 | 5 |

这 500 个画面描述题只覆盖镜头与发色，**没有测其他属性的视觉选源**。不能把当前编辑任务涉及服装、身份等理解为已经评估了这些属性的历史选择。

16 个错误中，7 个调用过额外查看历史，9 个没有。它们的选择工具调用格式均合法，没有因非法编号或格式失败而被记错。全测试共 3040 次回答中的 8 次格式错误已按既定规则纠正，最终没有失败题。

本次读取了全部 16 题的冻结公开输入、私有答案、实际内容卡、选择结果和查看历史记录。私有答案只用于这次事后检查，没有送回模型。挑出每类现象中字典序最小的代表题，CPU 提取标签候选与实际选择候选的第 1、8 秒帧；没有调用模型、修改标签或调整提示词。

## 错误中实际看到了什么

下面的分类是排查现象，不是已经证明的模型内部原因。模型只输出工具调用，没有输出选择理由。

| 现象 | 题数 | 解释 |
|---|---:|---|
| 卡片漏掉前后、侧向差别 | 3 | 候选卡把不同朝向写得相似，或把标签要求的正面写成背面。 |
| 卡片已经有方向线索，但仍选另一版 | 2 | 例如一张卡明确写“背后偏左”，仍选了单纯背面版。 |
| 卡片漏掉推进、拉远信息 | 4 | 动态镜头被写成静止，或没有同时说清方向和距离变化。 |
| “低角度全身”可匹配多个候选 | 2 | 题面没有说明是否静止，其他环绕镜头也可能符合这个宽泛说法。 |
| 卡片中发色难以区分 | 5 | 要找粉发/红发，但候选都被写成黑或深棕；重新看视频也不保证分清。 |

## 五个代表题的画面核对

1. **0000628，正背面与后侧面被概括成相同方向。** 标签 0、选择 1。实际帧里 0 是正背面，1 是后侧面；两张卡却都写了 behind，其中 1 写了 full-body，更容易与题面字面贴近。历史记录也写明 1 是 rear three-quarter，所以不能只归咎于视频观察，选择时没有充分利用已有记录也可能参与。[视频0](understanding_review_frames/0000628_video0_1s.png) · [视频1](understanding_review_frames/0000628_video1_1s.png)

2. **0000715，有方向线索仍然选错。** 题目要后侧全身，标签 0、选择 1。帧显示 0 为后侧、1 为正背；卡0已写“behind and to the left”，却未被选中，也没有额外查看。这比单纯卡片缺信息更接近选择阶段的线索取舍问题，但仍不能还原模型具体理由。0000719 共享同一对视频，发生同样选择。[视频0](understanding_review_frames/0000715_video0_1s.png) · [视频1](understanding_review_frames/0000715_video1_1s.png)

3. **0001560，卡片遗漏镜头拉远。** 题目要背面从上半身拉远到全身。视频0的1秒/8秒确实由近到远且始终背面；视频1同样拉远，但侧向拍摄。两张卡都说 static，0虽保留背面信息，1未注明侧面。这里既有观察摘要遗漏，也有选源没有抓住方向差别。[视频0第1秒](understanding_review_frames/0001560_video0_1s.png) · [第8秒](understanding_review_frames/0001560_video0_8s.png) · [视频1第1秒](understanding_review_frames/0001560_video1_1s.png) · [第8秒](understanding_review_frames/0001560_video1_8s.png)

4. **0000678，粉发在渲染画面中并不显眼。** 标签0为粉发、1为改黑发，卡片均写Black；模型额外看了0/1，仍选1。对照帧中，0有暗红粉色反光，1更接近黑色，但人物和背景受强烈红色霓虹照明，0并不是容易识别的浅粉色头发。这支持“可见颜色差异弱/观察归纳不准确”的解释，但不能据此宣布标签错误，也不能把其他4个发色错误都归因于同一种光照。[视频0](understanding_review_frames/0000678_video0_1s.png) · [视频1](understanding_review_frames/0000678_video1_1s.png)

5. **0001161，题面本身可能不够唯一。** 题目只说low-angle full-body。标签0是基本固定取景，选择2的背景随时间变化，但两者都能看到低角度全身。题面没写static，不能仅凭内部镜头预设类别认定只有0满足字面要求。模型查看了1、2、3，却没查看标签0。此例保留原评分，报告潜在多解；0004794在卡片文字上也有类似现象，但没有人工逐帧检查它。[视频0](understanding_review_frames/0001161_video0_1s.png) · [视频2第1秒](understanding_review_frames/0001161_video2_1s.png) · [第8秒](understanding_review_frames/0001161_video2_8s.png)

## 全部 16 题逐例记录

编号写作“标签→实际选择”。“查看”列列出模型额外查看的视频编号，横线表示未调用。JSON证据中保存完整原文、所有历史卡、私有答案、原始工具调用、文件路径与校验值：[完整记录](understanding_error_review.json)。

| 样本 | 属性 | 编号 | 查看 | 观察到的现象 |
|---|---|---|---|---|
| [0000628](../formal_v1/understanding/cases/selection_v3_test_0000628/decision/edit_request.json) | 镜头 | 0→1 | — | 卡0写behind但称medium-long；卡1也写behind且称full-body，遗漏后侧角度。历史编辑文字明说1是rear three-quarter，仍选1，可能同时涉及卡片概括和线索取舍。 |
| [0000678](../formal_v1/understanding/cases/selection_v3_test_0000678/decision/edit_request.json) | 发色 | 0→1 | 0,1 | 要求pink，卡0和1都写Black。回看0/1后仍选1；历史文字说1改成black，模型并未据此选0。 |
| [0000715](../formal_v1/understanding/cases/selection_v3_test_0000715/decision/edit_request.json) | 镜头 | 0→1 | — | 卡0已有behind and to the left，卡1只有behind；题目要求rear three-quarter，却选1，且未回看。卡片用medium-long/full-body不同措辞可能干扰；不能从仅工具输出还原内部原因。 |
| [0000719](../formal_v1/understanding/cases/selection_v3_test_0000719/decision/edit_request.json) | 镜头 | 0→1 | — | 与0715共享0/1两视频，另有第2个正面近景版本。卡0已有后侧方向，仍选1，未回看。 |
| [0001161](../formal_v1/understanding/cases/selection_v3_test_0001161/decision/edit_request.json) | 镜头 | 0→2 | 1,2,3 | 要求low-angle full-body。所有候选卡都含低角度全身，区别主要在静止/移动。回看[1,2,3]，未回看标签0，最后选2。 |
| [0001560](../formal_v1/understanding/cases/selection_v3_test_0001560/decision/edit_request.json) | 镜头 | 0→1 | — | 要求rear dolly-out。卡0有rear但写static；卡1也写static且未说明侧向，历史文字说1为side-view dolly-out。未回看，选1。 |
| [0001563](../formal_v1/understanding/cases/selection_v3_test_0001563/decision/edit_request.json) | 镜头 | 0→1 | — | 要求rear dolly-out；卡0写static背面，卡1写static又写slowly pulls back，方向不明。选择1，未回看。 |
| [0001568](../formal_v1/understanding/cases/selection_v3_test_0001568/decision/edit_request.json) | 镜头 | 0→1 | — | 要求rear dolly-out；卡0背面static，卡1static且无方向；历史文字把1写作side-view dolly-out。选择1，未回看。 |
| [0001950](../formal_v1/understanding/cases/selection_v3_test_0001950/decision/edit_request.json) | 镜头 | 2→1 | — | 标签2应匹配front three-quarter，卡2却写behind；卡1未写朝向，只写medium-full/low angle，模型选1且未回看。历史文字明说2改为front three-quarter。 |
| [0002003](../formal_v1/understanding/cases/selection_v3_test_0002003/decision/edit_request.json) | 镜头 | 0→1 | 0,1 | 要求front three-quarter；标签0卡写behind，卡1不写朝向。回看0/1后选1；历史文字说1改为behind。 |
| [0003334](../formal_v1/understanding/cases/selection_v3_test_0003334/decision/edit_request.json) | 镜头 | 0→1 | — | 要求side dolly-in；卡0有slowly zooms in但未写side，卡1static medium shot也未写side。历史文字说1是侧面上半身，模型选1，未回看。 |
| [0004318](../formal_v1/understanding/cases/selection_v3_test_0004318/decision/edit_request.json) | 发色 | 0→2 | — | 要求red；标签0卡为dark brown/black，卡2为reddish-brown/auburn，因此选2与当前卡文字更贴近。未回看；1的历史文字说red改pink，后续仅改场景。 |
| [0004457](../formal_v1/understanding/cases/selection_v3_test_0004457/decision/edit_request.json) | 发色 | 0→1 | 0,1 | 要求pink；0/1卡都为dark brown/black，回看后选1；历史文字说1改red。 |
| [0004484](../formal_v1/understanding/cases/selection_v3_test_0004484/decision/edit_request.json) | 发色 | 0→2 | 0,1,2 | 要求pink；标签0卡dark brown/black，候选1 reddish-brown、2 red，其他也暗色。回看0/1/2后选2。 |
| [0004493](../formal_v1/understanding/cases/selection_v3_test_0004493/decision/edit_request.json) | 发色 | 0→1 | 0,1 | 要求pink；0/1均写dark brown/black，回看后选1；历史文字说1改red。 |
| [0004794](../formal_v1/understanding/cases/selection_v3_test_0004794/decision/edit_request.json) | 镜头 | 0→1 | 0,1 | 要求low-angle full-body，0/1两张卡均明确满足；0static，1角色旋转/历史记录为360°orbit。回看后选1。 |


## 能得出的结论与不能得出的结论

这些错误集中在细粒度镜头关系和可见发色辨认。内容卡确实可能丢失区分信息；但0715这类题已经有线索仍选错，不能把所有错误都解释成信息卡不足。额外查看过的7题也没有全部恢复正确，说明重新看视频不是保证。

这次理解测试使用现有真实历史视频，不是生成后逐轮回填，因此不能把错误笼统归因为生成漂移。另一方面，数据标签来自构建时的属性和镜头设置，实际成片还受光照、人物姿态、取景运动和参考描述宽泛程度影响。984/1000保持原冻结评分；本分析不删除难例、不改答案，也不声称低角度歧义题已经由人工正式改判。

抽查只看了5个代表题的两时间点，其中0715的媒体也覆盖0719的同一对候选。其余10题没有人工看帧，原因判断仅来自实际卡片与记录。两帧也不能替代完整视频运动审核。本报告是事后错误描述，不是新一轮调参或正式人工重标注；全局文档、冻结代码、配置、数据、模型和作业均未改动。

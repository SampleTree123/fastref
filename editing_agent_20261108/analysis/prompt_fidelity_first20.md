# 前 20 个已完成样本：编辑提示词是否保留原意

## 2026-11-08T02:33:54+08:00：只读抽查结果

取样时间：2026-11-08T02:32:28+08:00。当时有 57 个理解样本完成，将完成样本按 sample_id 排序，固定取前 20 个。后续新完成的题不替换这份名单。本次只比较用户原话、参考列表和实际发给编辑器的文字，不评价选源编号，也不改正式运行内容。

这 20 例中，19 例未发现明显文字保真问题；1 例存在轻微风险：把用于寻找历史视频的镜头描述带入了生成要求。全部参考编号及顺序一致，未发现改掉目标发色/动作/衣物、漏掉明确保持要求或颠倒正背视图的情况。这是小范围人工式检查，不是 1000 题的整体正确率，也不代表生成视频效果通过。

覆盖：发型 5、发色 4、服装 5、动作 3、场景 3。没有人物身份替换或镜头编辑目标；题目中出现的镜头描述只是历史选择线索。16 例带参考，共 26 个参考项；4 例纯文字改发色。

## 需要注意的一例

`selection_v3_test_0000284` 的用户先说从“正面向前推进、全身逐渐变上半身”的历史视频开始，再要求红发改为栗棕色。输出保留了栗棕目标，但添加了：

> Preserve ... the front dolly-in camera movement from full-body to upper-body framing

这句话来自历史选择条件，并非新的编辑目标。保持源视频镜头本身合理，但把该具体描述重新交给生成器，在历史视频实际画面与描述有偏差时可能形成额外约束。这里只记录风险；没有运行该题的视频生成，不能断言已经造成错误。相对地，`0000282` 和 `0002118` 中类似的镜头定位描述没有被复制进去。

## 几个有代表性的正确保留

- `0000582`：删除“第 4 次编辑之前的输入”定位句，仍允许修改新环境及其对人物的光照，同时保留身份、动作和镜头。
- `0001235`：完整套装仍包括上衣、裙子和鞋；保留“移除不属于目标套装的原衣物”，并保留第 1 张正面、第 2 张背面。
- `0000678`：目标仍为参考中的思考站姿，明确不复制参考人物外貌、场景或镜头。
- `0000419`：只换上衣，没有扩大到整套。补充设计、颜色、图案和版型与参考一致，未指定参考之外的新颜色或款式。

## 逐例记录

完整原文、实际编辑要求、参考元数据、文件校验值和逐项判断都保存在[JSON证据](prompt_fidelity_first20.json)。下表中的链接指向当时实际完成的结果文件。

| 样本 | 编辑内容 | 判断与证据 |
|---|---|---|
| [0000152](../formal_v1/understanding/cases/selection_v3_test_0000152/worker_result.json) | 发型 | 去掉黑发版本选择句；发型匹配与保持脸、发色、服装和其余内容全部保留。 |
| [0000225](../formal_v1/understanding/cases/selection_v3_test_0000225/worker_result.json) | 发色 | 去掉第2次编辑结果的定位；黑发→白发保留。新增保持身份和发型，符合仅改发色的要求。 |
| [0000282](../formal_v1/understanding/cases/selection_v3_test_0000282/worker_result.json) | 发型 | 去掉前推镜头定位句；只改参考发型并保留身份、发色、服装等，没有把镜头选择描述复制进来。 |
| [0000284](../formal_v1/understanding/cases/selection_v3_test_0000284/worker_result.json) | 发色 | 红发→栗棕色目标保留，但把选源用的前推镜头、全身到上半身描述复制为明确保持要求。若实际选中画面存在偏差，这会给生成器额外限定；尚未观察该题生成效果，不能说已造成错误编辑。 |
| [0000419](../formal_v1/understanding/cases/selection_v3_test_0000419/worker_result.json) | 服装 | 只换上衣，其余保持；补充正背视图对应及设计、颜色、图案、版型匹配参考，没有凭空指定新款式或颜色。 |
| [0000432](../formal_v1/understanding/cases/selection_v3_test_0000432/worker_result.json) | 服装 | 去掉红发版本定位；保留上衣、下装、鞋，完整替换而非叠加，以及第1张正面、第2张背面的对应。 |
| [0000540](../formal_v1/understanding/cases/selection_v3_test_0000540/worker_result.json) | 动作 | 去掉最新版本定位；动作来自参考视频，身份、头发、服装、表情、场景和镜头保持。 |
| [0000582](../formal_v1/understanding/cases/selection_v3_test_0000582/worker_result.json) | 场景 | 去掉第4次编辑前输入的定位；环境及人物受光可以改变，身份、动作、镜头等保持，未把光照全部锁死。 |
| [0000655](../formal_v1/understanding/cases/selection_v3_test_0000655/worker_result.json) | 动作 | 去掉第1次编辑前定位；向前走及只借用参考动作保留，人物/环境/镜头保持。 |
| [0000675](../formal_v1/understanding/cases/selection_v3_test_0000675/worker_result.json) | 场景 | 去掉黑发版本定位；背景取参考，人物光照随新环境调整，其他明确保持项完整。 |
| [0000678](../formal_v1/understanding/cases/selection_v3_test_0000678/worker_result.json) | 动作 | 去掉粉发版本定位；思考站姿与仅转移动作保留，不借用参考人物、场景或镜头。 |
| [0001235](../formal_v1/understanding/cases/selection_v3_test_0001235/worker_result.json) | 服装 | 保留上衣、裙子、鞋和去除不属于新套装的旧衣物；正背视图顺序完整，只调整句序。 |
| [0001611](../formal_v1/understanding/cases/selection_v3_test_0001611/worker_result.json) | 发型 | 去掉红发版本定位；发型参考与原发色保持均在，正背方向来自公开参考元数据。 |
| [0001683](../formal_v1/understanding/cases/selection_v3_test_0001683/worker_result.json) | 场景 | 去掉第2次编辑结果定位；保留人物身份、外观、动作、镜头，新场景内自然受光允许改变。 |
| [0001747](../formal_v1/understanding/cases/selection_v3_test_0001747/worker_result.json) | 服装 | 删除套话，完整编辑要求基本原样保留，包含整体替换而非增加衣物及正背顺序。 |
| [0001906](../formal_v1/understanding/cases/selection_v3_test_0001906/worker_result.json) | 发型 | 去掉第3次编辑前定位；只使用参考发型、保持身份/发色/服装全部保留，补充正背方向。 |
| [0001998](../formal_v1/understanding/cases/selection_v3_test_0001998/worker_result.json) | 服装 | 去掉最新版本定位；连体衣+鞋整体替换，其他保持；第1张正面、第2张背面保留。 |
| [0002115](../formal_v1/understanding/cases/selection_v3_test_0002115/worker_result.json) | 发色 | 去掉第2次编辑结果定位；栗棕→白色保留，新增保持发型和身份没有改变原目标。 |
| [0002118](../formal_v1/understanding/cases/selection_v3_test_0002118/worker_result.json) | 发色 | 去掉静态背面全身镜头定位；目标栗棕保留，仅泛化保持其他内容，没有抄入具体镜头要求。 |
| [0002136](../formal_v1/understanding/cases/selection_v3_test_0002136/worker_result.json) | 发型 | 去掉第1次编辑结果定位；只换发型、保持发色和其余内容完整，正背参考用途正确。 |

限制：完成顺序和排序都会影响这批样本，不能代表全部题型。没有查看参考像素、源视频或生成结果，也没有尝试把选源调整到标准答案。正式代码、提示词、数据和作业全部保持不变；本文件只记录已发生输出中的文字现象。

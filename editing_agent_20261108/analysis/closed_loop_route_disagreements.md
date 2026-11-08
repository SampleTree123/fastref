# 完整生成链的选源差异复核

## 2026-11-08T07:55:52+08:00：前60链唯一差异应保留为一次错误回溯

范围固定为当时已完成的前60条链、256轮。原计划路线一致255/256；普通继续编辑209/210，其余类型均一致。之后正在运行的100链扩展不在这次快照内。

差异位于 **test_chain_13582第5轮：应从最新视频4继续，实际选了旧视频2。** 这次用户没有要求回溯；不能因为旧视频更符合“金发”描述，就把这次选择改判正确。

### 当轮实际输入与选择

用户原话：

> Using the provided source video, apply the following edit: The person currently has blonde hair; please change it to black.

原中文是“请将人物当前的金色发色替换为黑色”。它描述当前编辑的前提和目标，没有“回到”“从那个金发版本开始”等历史选择要求。

冻结系统规则明确：没有回到旧版本的请求时，继续最高已有编号。该轮已有0–4，因此应选4。私有标签同样记录`normal_continue`、parent4，未发现标签把一条回溯指令错标为继续的证据。

模型第一次调用`inspect_history([1,2,3])`，第二次调用`select_source_video(2)`。两次格式均合法，没有纠正。最新视频4的文字卡和全部历史过程已经在模型输入中，但重新查看的视频不含4。原始输出只有工具调用，没有解释理由，因此**无法确定它为什么没有查看4**。

LTX实际读取的源路径及SHA与第2轮输出一致，说明这不是展示或统计错位：生成确实从2开始，绕过了第3轮新发型和第4轮身份编辑。

### 金发与黑发的冲突来自哪里

| 版本 | 历史卡描述 | 离线查看1/8秒实际画面 |
|---|---|---|
| 生成视频2 | Bright blonde，较短发型 | 明显金色短发 |
| 生成视频3 | Black，刘海与双低辫 | 明显黑色刘海、双低辫 |
| 生成视频4（最新） | Black，刘海与双低辫 | 仍是黑色刘海、双低辫 |
| 原数据GT视频4 | 不提供给选源模型 | 金色刘海、双低辫 |

第3轮本来只要求换发型，并保持发色；生成时却提前变成了黑发，第4轮继续保留。于是第5轮固定文字中的“当前是金发”与实际最新结果冲突。就检查的两个时点而言，卡片把3/4写成黑发是符合画面的，不是把金发误认黑发。

这提供了一个可信的诱因：模型可能把“当前金发”误当成寻找历史金发版本的条件，查看旧视频后选择了2。**这是对可见行为的解释，不是模型直接报告的思考过程。** 无论这个诱因是否成立，当前协议都没有授权自动回溯；应保持在4上处理本轮编辑，而不是偷偷撤销此前轮次。因此仍记录为默认继续规则下的选源错误，不修改255/256的统计。

### 对本次结论的含义

这例同时暴露两件事：先前编辑可能意外改变应保留的属性；当后续文字还沿用原数据的外观前提时，Qwen可能把普通编辑误解成历史检索。不能简单归结为“卡片识别错”，也不能因生成历史变化而替错误选源改标签。

只检查了实际视频2/3/4和GT4的1、8秒帧，足以支持这里的发色对比；不代表完整运动或身份质量验收。本轮没有修改任何模型、提示词、标签或重新生成结果。

### 证据

- [第5轮第一次选源调用](../formal_v1/closed_loop/cases/test_chain_13582/turn_05/select_call_00.json)：包含全部历史卡/过程、当轮原话、inspect参数。
- [查看历史后的选源调用](../formal_v1/closed_loop/cases/test_chain_13582/turn_05/select_call_01.json)：实际看1/2/3，随后选2。
- [实际编辑要求](../formal_v1/closed_loop/cases/test_chain_13582/turn_05/edit_request.json)；[实际生成progress](../formal_v1/closed_loop/cases/test_chain_13582/turn_05/progress.json)。
- [冻结公开链](../formal_v1/frozen_sources/qwen3.8/datasets/editing_agent_v1/chains_test.jsonl)；[冻结私有标签，仅离线审查](../formal_v1/frozen_sources/qwen3.8/datasets/editing_agent_v1/private/chains_test_labels.jsonl)。
- 生成2：[1秒](closed_loop_route_frames/13582_actual_v2_1s.png)、[8秒](closed_loop_route_frames/13582_actual_v2_8s.png)；生成3：[1秒](closed_loop_route_frames/13582_actual_v3_1s.png)、[8秒](closed_loop_route_frames/13582_actual_v3_8s.png)；生成4：[1秒](closed_loop_route_frames/13582_actual_v4_1s.png)、[8秒](closed_loop_route_frames/13582_actual_v4_8s.png)。
- GT4仅离线：[1秒](closed_loop_route_frames/13582_gt_v4_1s.png)、[8秒](closed_loop_route_frames/13582_gt_v4_8s.png)。
- [本次60链计数快照、原文、实际路径和SHA](closed_loop_route_disagreements.json)。动态report可能随后被100链结果更新，本JSON保留此次范围。

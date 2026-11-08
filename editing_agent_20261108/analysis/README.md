# 历史查看与工具调用统计

## 2026-11-08 11:50 SGT：全部评估已完成

理解测试1000题、完整生成128链543轮均已结束。最终结果从这里查看：

- [本轮总报告](../final_report.md)：方法、结果与已知问题。
- [128链生成统计](formal_v1_closed_loop_128/README.md)：真实生成历史的使用、回溯次数和耗时。
- [工具调用统计](formal_v1_closed_loop_128/tool_calls/README.md)：2314次调用、12次纠正、2次主动查看历史。
- [全量流程检查](formal_v1_closed_loop_128/runtime_audit.json)和[独立完成审查](completion_audit.md)：检查范围与逐项证据。
- [完整展示](../formal_v1/closed_loop/report/index.html)：包含所有视频；下载时保留整个report文件夹。

下方较早日期和30链命令是当时的统计示例。重新统计最终结果时，生成统计使用`--limit 128 --output analysis/formal_v1_closed_loop_128`；调用统计使用完整`chains_test.jsonl`、对应private标签及`--output analysis/formal_v1_closed_loop_128/tool_calls`。这些操作只读原始结果，不重新推理。

2026-11-08 03:43（新加坡）：1000题正式理解测试已全部完成，最终分组、方法和实际耗时见[最终报告](formal_v1_understanding/final_report.md)。下方02:30记录仅为过程快照。

这里只读取已保存的运行结果，不调用模型、不修改冻结的代码或数据。运行中的报告是当时的快照；等任务完成后再运行一次即可得到完整统计。

`summarize_runtime.py` 检查模型实际额外查看了多少次历史视频、各阶段是否需要纠正输出、最终有没有因输出不合法而终止，并记录所选编号和已有题型、历史卡片的重复引用、调用耗时。详细事件保留路径，方便回查原始记录。`test_summarize_runtime.py` 用两组小样验证统计：查看历史后的下一次选择不会误计为格式纠正；尚在进行的纠正不会误计为终止失败。

在本实验目录执行：

```bash
python analysis/test_summarize_runtime.py

python analysis/summarize_runtime.py \
  --run development_v3 \
  --dataset ../../datasets/editing_agent_v1/chains_debug.jsonl \
  --labels ../../datasets/editing_agent_v1/private/chains_debug_labels.jsonl \
  --output analysis/development_v3

python analysis/summarize_runtime.py \
  --run formal_v1/understanding \
  --dataset ../../datasets/editing_agent_v1/understanding_test.jsonl \
  --labels ../../datasets/editing_agent_v1/private/understanding_test_labels.jsonl \
  --output analysis/formal_v1_understanding
```

每次输出一份中文 `README.md` 和一份包含事件与耗时明细的 `summary.json`。`--labels` 仅用于事后按题型统计，绝不传给模型；不提供时题型显示 `not_provided`。

几点读法：

- `inspect_history` 是模型主动额外查看历史，不包括程序最初建立视频卡片的查看。次数来自有效工具调用；视频槽位数是这些调用要求查看的视频数量之和，同一视频可能被多次查看。
- 输出编号分为只有原视频可选、最新视频、原视频、较早的其他视频。题型来自答案文件，没有通过 prompt 关键词猜测。
- 校验失败包括 JSON、字段或参数不合要求。只在最终错误记录明确说明时才统计为该阶段终止失败。选错合法编号并不属于格式错误。
- 卡片重复引用只能说明已保存历史共用了观察结果，不等于精确缓存命中次数。没有观察标识的卡片不参与该项统计；这不表示它没有被观察。
- 调用耗时来自实际记录。多卡工作有重叠，不能把这些秒数相加当作整个作业用时；等待共享缓存的时间也不能当成模型计算时间。
- 这里不自动打视频质量分，也不能代替人工验收。

2026-11-08 02:30（新加坡）：开发版 6 条链、26 轮完成；111 次模型调用均有效，主动查看历史 1 次，查看了 2 个视频。正式理解测试仍在执行，快照仅反映当时已落盘内容，见各报告时间。

## 核对1000题的实际输入输出

`audit_understanding.py` 核对编号、参考顺序、历史视频查看和信息卡来源。它读取固定输入及已保存的模型调用，不读取正确答案，也不调用模型。

```bash
python -B analysis/audit_understanding.py
python -m unittest discover -s analysis -p test_runtime_audit.py -q
```

结果在 `understanding_runtime_audit.json`。完成后应看到 `audit_passed=1000`、`audit_failed=0`；若有题目运行失败，则单独报告原因，不能把失败或待运行的题目算作检查通过。8项篡改日志测试用于确认检查能发现编号、参考或信息卡被改错。

## 完整生成链的阶段与耗时

`summarize_generations.py` 单独统计选源、成功生成、完成观察并写入历史三个阶段。它读取每轮实际LTX收据，跨资源接续后也不会把worker本次启动耗时误当整链耗时。源类型通过实际请求路径与本链root/之前生成文件对应；回溯与继续按当时已有编号判断。文件SHA的严格核验仍由独立runtime审查负责。

```bash
python analysis/test_summarize_generations.py
python analysis/summarize_generations.py \
  --run formal_v1/closed_loop \
  --dataset ../../datasets/editing_agent_v1/chains_test.jsonl \
  --limit 30 \
  --output analysis/formal_v1_closed_loop_30
```

四个小样分别检查：已出视频但尚未观察记录、已选源但还未生成、完整链在续跑后的计数与实际耗时，以及失败worker加上较新的失败state仍计为失败。统计不从文件修改时间猜测恢复；状态冲突另列问题。脚本不调用GPU、不改原始结果，也不读取模型自评作为质量分。显存数字只覆盖去噪开始至解码结束阶段，不包含之前编码的峰值。

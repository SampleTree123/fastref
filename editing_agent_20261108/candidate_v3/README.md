# 候选 v3：把视频观察与效果比较分开

这里是候选文件，没有改动正在运行的 v2。

新生成视频先单独交给 Qwen。它看不到源视频、用户请求或参考，只记录这个视频自身的七类内容。内容卡保存后，再进行另一轮比较：读取源视频、新视频、固定内容卡、用户原话和实际执行的编辑要求，以及本轮参考，记录真正改变了什么。用户原话中的历史选择已完成，不能拿原话里的外观描述当作看到的事实。

比较阶段只输出五个工具参数，不再输出或改写内容卡。卡片有按视频内容、模型与观察提示词确定的缓存，比较中断后继续不会重复观察。

14 项 CPU 检查通过，覆盖输入隔离、保持原请求、卡片缓存、中断恢复、严格输出格式。实际 Qwen 模板渲染、关闭思考和工具格式往返检查也通过。尚未在 GPU 上验证候选。

切换时只复制 manifest.json 的 promote_only 四个文件到 qwen3.8 中同名位置。observe/select/edit 的副本仅用于独立候选测试，内容未改。文件内部 ROOT 仍按原部署位置计算，无候选路径硬编码。建议在新输出目录运行，不能把 v2 的旧状态当作 v3 恢复。

候选 CPU 检查命令（从 DataMake 根目录执行）：

```bash
PYTHONPATH="$PWD/10_research/model_experiments/qwen3.8/scripts/evaluation" \
python -m unittest discover \
  -s 10_research/model_experiments/qwen3.8/experiments/editing_agent_20261108/candidate_v3/scripts/editing_agent \
  -p test_harness.py -v
```

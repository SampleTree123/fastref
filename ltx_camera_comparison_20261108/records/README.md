# 27例、四组最终模型对比

## 2026-11-08 21:41 SGT：全部生成并检查通过

四组均正常结束，实际生成100条，复用8条，展示108个结果位置。100条生成视频完整解码、帧数、尺寸和播放格式均通过；页面含179个实际媒体文件、27个案例、没有待生成占位。正在转移到fastref，以下是执行过程记录。

## 2026-11-08 21:10 SGT：正在生成

用户确认后，使用195783的八张H200进行推理。四组都使用连续时间编码、套装正背分开图、镜头参考图，最终768步权重。当前任务只生成结果，不训练。

| 组 | 训练时参考丢弃 | 参考CFG | 文本CFG | STG | Rescale | 负面提示词 |
|---|---|---|---|---|---|---|
| ① | 10% | 3 | 3 | 1 | 0.7 | 使用 |
| ② | 10% | 1 | 3 | 1 | 0.7 | 使用 |
| ③ | 10% | 3 | 1 | 0 | 0 | 不使用 |
| ④ | 无 | 1 | 3 | 1 | 0.7 | 使用 |

①②③使用193151最终权重；④使用191904最终权重。这里的参考CFG控制推理时参考内容的影响，不是训练时随机丢弃的概率。全部40步、seed42、12fps，横屏768×448、竖屏448×768；UE121帧、CoinVE57帧。720p尚未用于此次推理。

27例包括22例合成数据、5例CoinVE；其中4例身份替换是训练集展示，其余为测试集。原视频、真值、参考内容和四组输出放同一页面。

旧的单轮结果使用不同检查点，不能替代本次结果。8例没有额外参考内容，①②组在这些例子上的计算相同，因此共用②的输出。实际新生成100条，展示108个结果位置；复用的来源写进记录。

完成后将完整网页和实际媒体放进fastref，逐文件核对后删除本批LTX目录中的重复媒体。其他历史推理目录暂不删除；195783不自动释放。

## 文件用途与操作

- `campaign.json`：四组检查点、参数、输入、实际命令及GPU分配。
- `inputs/`：固定的27例名单、需要单独生成的19例名单、四组参数。
- `execution_state.json`：当前执行状态。
- 四个英文组名目录：生成视频、逐卡记录和总记录。
- `prior_result_check.json`：为什么旧结果不能直接使用，以及本次8例复用依据。
- `display/`：生成完毕后的完整离线网页；导出前不要把`gallery/`当独立展示包。

本批临时执行脚本由此文档维护，任务完成后作为复现入口保留。项目根目录执行：

```bash
srun --jobid=195783 --overlap --ntasks=1 --cpus-per-task=64 --gres=gpu:8 \
  bash scripts/common/run_trainer_python.sh scripts/inference/run_camera_comparison_campaign.py \
  --campaign experiments/inference/final_camera_comparison_20261108/campaign.json
```

脚本把GPU分为两组：前四卡依次生成②①，后四卡依次生成④③。它调用`run_fixed_cases_parallel.py`，再逐卡调用`run_fixed_case_inference.py`，之后进入原生LTX推理代码。已有结果目录不能重复启动。

生成全部结束后：

```bash
bash scripts/common/run_trainer_python.sh scripts/inference/finish_camera_comparison.py \
  --campaign experiments/inference/final_camera_comparison_20261108/campaign.json
```

完成脚本核对四组结果、补入8例复用记录，再调用`build_experiment_comparison.py`生成页面，调用`export_guidance_gallery.py`导出真实媒体，并核对108个结果文件。它不会自动写入fastref，也不会删除数据；转移核对后再做清理。

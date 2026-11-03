# 两组最新第450步检查点推理

使用189814中的一张H200，先原部件多图，再套装正背分开，各16例。使用第150步同一批案例和参数：40步、seed42、CFG3、STG1、rescale0.7以及相同负面提示词。第450步检查点在启动时固定保留，训练不暂停。

`manifest.json`指定本轮两组推理；`gallery_manifest.json`另外引入已完成的第150步结果，页面按原多图150/450、套装150/450四列展示。旧结果不重复生成。

- `state.json`、`logs/`：本轮进度。
- `original/`、`outfit/`：本轮新生成的32条视频及记录。
- `gallery/index.html`：带旧结果的比较页，未完成部分显示待生成。
- `display/`：全部完成后导出的实际媒体和HTML，可整目录下载。

本轮入口：在项目根目录、空闲的189814资源内执行 `python experiments/inference/checkpoint_comparison/run_20261103_step450/run.py`。已启动时不要重复执行。调用路径：run.py → pin_for_validation固定检查点 → run_fixed_cases_parallel.py → run_fixed_case_inference.py → 官方原生推理；随后build_experiment_comparison.py比较四组，export_guidance_gallery.py导出。

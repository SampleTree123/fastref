# 本轮完成后的文档收尾建议

## 2026-11-08 11:48 SGT：本次采用的处理方式

主agent决定不修改治理或检查器，不新建后继计划。128链生成、最终运行/媒体检查及交付已经完成；R01—R07现均为completed，同时保留唯一current计划作结果入口，明确没有新增执行任务。计划文件暂不迁移，归档等下一份经授权计划确定后再处理；不声称已经正式归档。

下方是此前核查建议，治理空档调整方案本轮不采用。它不影响GPU评估或实际结果；最终审查与交付现已完成；保留current不表示仍在执行，保留路径也不表示已归档。


核对时间：2026-11-08T07:12:19+08:00。**下方为当时的执行建议与规则核对；实际处理以上方最新决定为准。未改治理或检查器，未迁移计划。**

## 先确认何时可以关闭

128条链跑完本身不等于收尾完成。应先核实实际成功/失败数量、完整运行审查、最终页面和报告，并把R06、R07的结果及范围写清楚。编辑质量、提示词提升等没有证明的结论仍标为未证明；不能把程序成功写成全面质量验收。达到原计划的交付条件后再关闭，不新增未经用户授权的实验或计划。

## 当前规则存在一个空档

[GOVERNANCE §7.2](../../../../../../20_planning/data_design/character_creator/full_process/GOVERNANCE.md)、计划README、Master和文档维护skill都要求“恰好一个current主计划”。当前PLAN_INDEX也没有后继授权计划。因此：**现行规则不允许当前计划为空；也不能让已完成计划冒充current来满足检查器。**

检查器实际入口`check_plan_subsystem()`第787行拒绝0个current；`check_status_and_plans()`第1082行有相同旧逻辑，但目前`audit_docs()`未调用后者。交叉检查核对`current_plan_id`；现有检查器未直接核对`current_plan_path`，这不意味着它可以指向completed而继续声称current。

已用临时测试树验证：原样通过；没有current并设置两个字段为null后，只出现`[Plan] expected exactly one current plan, found 0`。正式文件未改。

建议收尾时明确解决这个治理空档：**有授权活动任务时恰好一个current；全部完成且没有新授权时允许零个，并明确记录空闲原因。**这属于治理语义调整，应同步Governance、Master、计划README/active入口、维护skill和检查器，并按治理要求留决定记录；不是创建新的执行计划。若该规则尚未确认/更新，就诚实报告这项文档检查冲突，不伪造后继计划，也不掩盖失败。

## 能否暂时保留current作为结果入口

当前计划只有`生命周期：current`和`计划状态：current`两个元数据字段；没有已定义的独立`execution_status`。检查器明确要求前两个字段相同，因此**不能写生命周期current、计划状态completed**。需求表R01—R07可以全部completed而外层仍current，检查器不会因此报错；也可以另写“执行已完成，文档收尾/归档待决定”的普通说明，但这不是现有规则中的正式关闭。

因此可用的短期过渡是：任务实际完成后，把实验state和报告的执行状态据实记为completed；主计划暂留current作为唯一展示入口，开头明确“测试已完成，无运行任务；仅待文档归档决定，不执行新增工作”。PROJECT_STATE暂保留该计划ID/路径，phase和next_action明确等待查看/收尾。这个过渡不冒充GPU仍在执行，也无需制造后继计划，**但必须标为待归档，不能宣称已经满足正式计划关闭**。skill明确要求“move a fully complete plan to completed history”，Governance也要求迁移，所以长期保留并不符合其完整收尾要求。

当前仍在评估，以上连过渡状态都尚未实施。治理空档只是最终文档关闭问题，不阻碍GPU任务继续。最终采用短期过渡还是最小规则调整，由主agent在结果齐全后决定；本建议不授权修改治理或检查器。

## 关闭后的字段建议（仅在主agent决定并完成空档规则调整后）

`PROJECT_STATE.json`建议使用以下明确值；新增字段是建议，不是现有schema要求：

```json
{
  "current_plan_id": null,
  "current_plan_path": null,
  "plan_execution_status": "idle_no_authorized_plan",
  "last_completed_plan_id": "PLAN-2026-11-QWEN-LTX-EDITING-AGENT",
  "last_completed_plan_path": "project_management/history/completed_plans/PLAN-2026-11-QWEN-LTX-EDITING-AGENT.md",
  "project_phase": "qwen_ltx_evaluation_delivered_awaiting_user_review",
  "next_action": "本轮结果已交付，等待用户查看和决定下一步；不自动开始新实验。"
}
```

同步实际`updated_at`、`model_research_status`中的评价状态/作业终态/最终报告链接；保留“质量检查的范围”，不要清空无关UE Workflow或其他已接受结果。PROJECT_STATUS的“当前计划”写“无；本轮已完成，等待用户下一步”，另链接最近完成计划。PLAN_INDEX的Current写“暂无已授权执行的计划”，Completed新增本轮。

检查器改动应同时验证：零current+两个null+明确idle状态允许；零current却残留ID/路径拒绝；一个current时ID和路径必须对应；多个current拒绝；completed不能伪装current。旧入口若保留，也应同规则。不要只删掉“必须一个”的判断。

## 归档及链接清单

只有最终证据齐全后执行：

```bash
DOC=20_planning/data_design/character_creator/full_process
mv "$DOC/project_management/plans/active/PLAN-2026-11-QWEN-LTX-EDITING-AGENT.md" \
   "$DOC/project_management/history/completed_plans/PLAN-2026-11-QWEN-LTX-EDITING-AGENT.md"
```

- 计划的生命周期、计划状态均改`completed`；写关闭日期、R01—R07最终状态、证据和“未纳入/后续”范围。History检查器要求完成计划有这些范围说明。
- 更新PLAN_INDEX、plans/active/README、history/completed_plans/README；在MILESTONE_LOG顶部追加本轮事实和链接。历史目录README已有通用入口，无新变化可不改。
- 更新PROJECT_STATUS/PROJECT_STATE、HANDOFF、根README、模型进度总览及本实验README/state/final_report，消除“正在执行”的旧当前状态。历史日期段落保留原事实，只修链接并明确为当时状态。
- 原LTX计划中的继任链接需要指向归档后的本计划；迁移计划内部指向原LTX计划的`./PLAN-2026-10-LTX25-SINGLE-TURN.md`改为`../../plans/active/PLAN-2026-10-LTX25-SINGLE-TURN.md`。其他相对链接逐个验证；两目录深度相同不代表所有链接都无需改。
- 不改已经冻结的formal输入/代码快照，不迁移大型输出，不删除失败证据。

最后执行文档检查和回归；这是未来收尾命令，本次没有执行正式全库检查：

```bash
python -B 00_engine/render_pipeline/tools/documentation/check_character_factory_docs.py --json
python -m pytest 00_engine/render_pipeline/tools/documentation/tests -q
rg -n 'PLAN-2026-11-QWEN-LTX-EDITING-AGENT|恰好一个.*current|恰好一个.*当前主计划' \
  HANDOFF.md README.md 20_planning/data_design/character_creator/full_process \
  .codex/skills/character-factory-doc-maintenance/SKILL.md
```

本次只新增此建议文档，未移动计划、修改正式状态、改变冻结文件或调用GPU。

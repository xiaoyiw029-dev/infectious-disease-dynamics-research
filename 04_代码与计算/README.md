# 计算与结果索引

当前以周新增流量进行预测。以下是现有入口，不需要逐个运行全部脚本。

| 工作 | 脚本（00_脚本） | 结果位置（项目根目录下） |
|---|---|---|
| 周监测数据整理 | 01_prepare_weekly_cases.py | 02_数据/01_清洗数据 |
| 气候小时到周汇总 | 02_prepare_weekly_climate.py | 02_数据；详见数据质量记录 |
| 2024周编码核查 | 13_audit_week_fields_2024.py | 02_数据/04_数据质量控制 |
| 全国气候选择及缺口审查 | 14_select_main_climate_candidate.py、15_audit_climate_gap_weeks.py | 02_数据/01_清洗数据、04_数据质量控制 |
| 30周起训滚动验证 | 16_validate_baseline_rolling.py | 05_结果与图表/04_滚动验证/development_rolling_v01 |
| 峰前补充和趋势对照 | 17_validate_peak_trend.py | 05_结果与图表/04_滚动验证/development_peak_trend_v01 |
| 统计气候增益筛查 | 18_screen_climate_increment.py | 05_结果与图表/04_滚动验证/climate_increment_screen_v01 |
| 峰前过冲诊断 | 19_diagnose_peak_overshoot.py | 05_结果与图表/04_滚动验证/peak_overshoot_diagnostics_v01 |
| 温度滞后输入准备 | 20_prepare_temperature_extension.py | 05_结果与图表/04_滚动验证/temperature_lag4_preparation_v01 |

## 依赖关系

16号入口调用08、09号模型实现；17号复用16号及已保存的30周起训结果；18、19号依赖前面的验证实现或结果。五组滚动验证目录均需保留，不能因时间较早而删除。

07、08、09号脚本仍可用于早期设置复算。其中09号独立执行默认固定250/500，正式A4由16号入口传入189/63。10、12号属于已停用的B组存量拟合和绘图，已随冗余输出清理；相关设置和数值见[研究过程记录](../09_项目管理/研究过程记录.md)。

## 检查与使用

先查看对应脚本的 `--help`，核实输入、输出目录及参数，再运行；不直接覆盖已确认结果。环境见[Python环境配置](02_环境配置/Python环境配置说明.md)。

在项目根目录运行已有测试：

```powershell
.\.venv\Scripts\python.exe -B tools\run_project_checks.py
```

此入口按各测试脚本要求分别传入项目和结果目录；这些脚本不能直接混用 `unittest discover` 的参数。汇报图优先查看扩展验证目录的 `expanded_forecasts.png`、`peak_forecasts.png`，以及过冲诊断目录的 `population_profile_diagnostics.png`。旧全段拟合与滚动预测的评分不可混用。

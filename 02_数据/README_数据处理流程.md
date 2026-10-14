# 数据处理流程

## 一、原始数据放置

数据Y数据：

```text
02_数据/00_原始数据/OpenDataSUS_SINAN_Chikungunya/
```

气候数据：

```text
02_数据/00_原始数据/inmet/
```

人口数据：

```text
02_数据/00_原始数据/ibge_population/
```

## 二、清洗数据输出

全国周数据Y序列：

```text
02_数据/01_清洗数据/brazil_weekly_chikungunya_cases.csv
```

全国周气候序列：

```text
02_数据/01_清洗数据/brazil_weekly_climate.csv
```

建模合并数据：

```text
02_数据/01_清洗数据/brazil_weekly_model_dataset.csv
```

## 三、第一阶段字段目标

建模数据集建议字段：

| 字段 | 含义 |
| --- | --- |
| year | 年份 |
| epi_week | 流行病学周 |
| week_start | 周开始日期 |
| cases | 全国周数据Y数 |
| population | 全国人口 |
| incidence_per_100k | 每 10 万人发病率 |
| temp_mean | 周平均温度 |
| rainfall_sum | 周累计降雨 |
| humidity_mean | 周平均相对湿度，可选 |
| temp_mean_lag1 | 温度滞后 1 周 |
| rainfall_sum_lag1 | 降雨滞后 1 周 |

## 四、处理顺序

1. 下载数据字典并确认字段。
2. 读取年度数据Y数据。
3. 按数据Y口径筛选。
4. 按全国流行病学周聚合数据Y。
5. 读取 INMET 气候数据。
6. 按周聚合全国气候指标。
7. 合并数据Y、气候和人口。
8. 输出建模数据集。

## 五、当前待填字段

数据Y数据核心字段已确认：

| 字段 | 含义 |
| --- | --- |
| `DT_SIN_PRI` | 症状开始日期 |
| `SEM_PRI` | 症状开始流行病学周 |
| `DT_NOTIFIC` | 通知日期 |
| `SEM_NOT` | 通知流行病学周 |
| `SG_UF` | 居住州 |
| `ID_MN_RESI` | 居住市镇 |
| `CLASSI_FIN` | 最终数据Y分类 |
| `CRITERIO` | 确认标准 |

第一版全国周数据Y序列已生成：

```text
02_数据/01_清洗数据/brazil_weekly_chikungunya_cases.csv
```

构建记录见：

```text
02_数据/全国周数据Y时间序列构建记录_v01.md
```

## 六、气候周数据现状（2026-10-05）

已取得 INMET 2023-2025 年自动站年度文件，先将 UTC 小时换算为站点当地日期，再以与数据Y相同的星期日—星期六周界构建 156 周气候表。当前 63 周拟合窗口的周键全部匹配。2026-10-05 确认以 2024 年 IBGE 人口加权州均值作为全国气候主候选，降雨优先使用完整 168 小时站点周；等州、等站点及 160 小时降雨版本仍保留作对照。UTC 日历日期初版保存在 `02_数据/02_中间数据/inmet_utc_reference_2023_2025/`，作为历史核查参照。

处理脚本：`04_代码与计算/00_脚本/02_prepare_weekly_climate.py`。

质量核查：`02_数据/04_数据质量控制/气候周数据准备与质量核查_2023-2025_v01.md`。

周表：`02_数据/01_清洗数据/brazil_weekly_climate.csv`；拟合窗口对齐表：`02_数据/01_清洗数据/baseline_climate_case_alignment_2023W35_2024W45.csv`。

主候选另存为 `02_数据/01_清洗数据/brazil_weekly_climate_main_candidate_2023_2025.csv`；选择与复核脚本为 `04_代码与计算/00_脚本/14_select_main_climate_candidate.py`，来源和校验记录为 `02_数据/01_清洗数据/climate_main_candidate_selection_v01.json`。原周表的通用列仍为等州汇总，不能将其误作已选人口加权版本。

2025-W52 的完整降雨仅覆盖 1 个 UF，不能作为全国输入；W43 覆盖也明显下降。两周经独立复算，缺口确认来自原始文件的空白字段，非本轮查出的汇总差异。数据Y周历中 2025 年有 53 周，W53 现已通过 2026 年小时资料补入，完整雨量覆盖 25 个 UF，有效州人口占比 98.841%。

当前主候选为 `02_数据/01_清洗数据/brazil_weekly_climate_main_candidate_2023_2025_v02.csv`，共 157 周；初版 156 周保留，其数值未改变。新程序 `15_audit_climate_gap_weeks.py` 负责目标周独立核查和跨年补齐，记录保存在 `02_数据/02_中间数据/inmet_2025_gap_audit/`，详细说明见 `02_数据/04_数据质量控制/2025年降雨缺口与跨年周核查_v01.md`。原 `02_prepare_weekly_climate.py` 与 `14_select_main_climate_candidate.py` 仍重现历史 156 周版，不应将其输出误作 v02。

主候选表的日历连续不等于全年可直接入模。训练与验证划分尚待确认，见 `03_模型与方法/01_参数估计/训练与验证划分方案_v01.md`；2025 年是否曾参与选模仍不确定。

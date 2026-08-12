# 数据处理流程

## 一、原始数据放置

病例数据：

```text
02_数据/00_原始数据/opendatasus_chikungunya/
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

全国周病例序列：

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
| cases | 全国周病例数 |
| population | 全国人口 |
| incidence_per_100k | 每 10 万人发病率 |
| temp_mean | 周平均温度 |
| rainfall_sum | 周累计降雨 |
| humidity_mean | 周平均相对湿度，可选 |
| temp_mean_lag1 | 温度滞后 1 周 |
| rainfall_sum_lag1 | 降雨滞后 1 周 |

## 四、处理顺序

1. 下载数据字典并确认字段。
2. 读取年度病例数据。
3. 按病例口径筛选。
4. 按全国流行病学周聚合病例。
5. 读取 INMET 气候数据。
6. 按周聚合全国气候指标。
7. 合并病例、气候和人口。
8. 输出建模数据集。

## 五、当前待填字段

由于尚未读取 OPENDATASUS 数据字典，以下字段名暂不能写死：

- 症状开始日期字段。
- 通知日期字段。
- 流行病学周字段。
- 病例分类字段。
- 居住地字段。

脚本中会以配置变量形式保留待填项。

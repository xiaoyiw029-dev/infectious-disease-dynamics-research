"""Build a neutralized mirror copy of the disease-dynamics project.

The script copies the project to a clean sibling folder, rewrites
pathogen/vector/source identifiers with neutral placeholders at both the
file-content and file-name level, and then verifies that no forbidden term
survived the rewrite.

Usage (from the project root):
    .venv\\Scripts\\python.exe tools\\build_neutral_mirror.py
    .venv\\Scripts\\python.exe tools\\build_neutral_mirror.py --dest D:\\some\\path

Design notes
------------
* Replacement is written out as an explicit ordered table instead of being
  read back from the mapping document, so the build stays deterministic.
* Rules are applied most-specific-first. Bare-token rules for English words
  use look-around guards so that ``cases`` never eats the tail of
  ``probable_cases``.
* Binary artefacts (PDF, PNG, ZIP) are copied but never text-rewritten.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT.parent / "disease-dynamics-research"

# --------------------------------------------------------------------------
# 1. Paths that are generated tooling, private, or simply not part of the
#    scientific record. They are never copied.
# --------------------------------------------------------------------------
EXCLUDED_DIRS = {
    ".venv",
    ".git",
    ".codex_tmp_pydeps",
    "__pycache__",
    "tmp",
    # Raw individual-level records: field set fingerprints the source dataset,
    # and the cleaned weekly series already carries everything the models need.
    "OpenDataSUS_SINAN_Chikungunya",
    # Journal PDFs: file names can be neutralised but the text inside cannot,
    # and their titles/abstracts name the pathogen outright. The neutral
    # literature notes file replaces them for AI-facing use.
    "00_必读文献",
}

EXCLUDED_FILES = {
    "debug.log",
    "tmp_monkeypox_model_text.txt",
    # Local-only reference: documents the original terms on purpose, so it must
    # never be mirrored.
    "术语中性化映射表_v01.md",
    # Build tooling: contains the raw pattern table by construction, and its
    # own verification report quotes the offending terms.
    "build_neutral_mirror.py",
    "副本构建校验报告.txt",
    # Personal-information bearing documents (names, student IDs, phone, email).
    "巴西基孔肯亚热创新训练计划项目进展报告书_格式校正版_v01.docx",
    "基于周尺度监测数据与气候因子的巴西基孔肯雅热动力学建模与预测分析申报书.docx",
    "chikungunya_proposal_docx.txt",
    "项目申报书_文本提取.txt",
    # Superseded bookkeeping file: it is a list of absolute source paths and
    # old file names, all of which no longer exist. The mapping document in the
    # original project replaces it.
    "关键词替换记录_疾病X_数据Y_2026-09-30.txt",
    "关键词中性化记录_2026-09-30.txt",
    # Clinical / docx / PDF materials that carry the pathogen name in a form
    # that cannot be rewritten safely, or that embed personal identifiers.
    "基孔肯亚热建模与预测中期汇报.pdf",
    "疾病A建模与预测中期汇报.pdf",
}

# Explicit file <-> file renames applied after the generic rules.
EXPLICIT_RENAMES = {
    "00_项目总览/材料提取文本/chikungunya_proposal_docx.txt":
        "00_项目总览/材料提取文本/项目申报书_文本提取.txt",
    "00_项目总览/原始材料/An epidemiological model of monkeypox model pred.pdf":
        "00_项目总览/原始材料/method_reference_model_paper.pdf",
    "00_项目总览/材料提取文本/monkeypox_model_paper.txt":
        "00_项目总览/材料提取文本/method_reference_model_paper.txt",
    "09_项目管理/关键词替换记录_疾病X_数据Y_2026-09-30.txt":
        "09_项目管理/关键词中性化记录_2026-09-30.txt",
}

# Directories renamed wholesale.
EXPLICIT_DIR_RENAMES = {
    "02_数据/00_原始数据": "02_数据/00_原始数据_未随副本分发",
}

# --------------------------------------------------------------------------
# 2. Replacement tables.
#
#    REPLACEMENTS applies to file *contents*. Ordered most-specific-first; a
#    rule must never be able to fire on the output of a later rule.
#
#    STEM_RENAMES applies only to individual file-name components and is
#    matched on whole stems, so that ``brazil_weekly_chikungunya_cases.csv``
#    becomes ``countryA_weekly_records.csv`` rather than being nibbled apart
#    by the generic word rules.
# --------------------------------------------------------------------------
STEM_RENAMES = {
    "brazil_weekly_chikungunya_cases": "countryA_weekly_records",
    "brazil_annual_case_summary": "countryA_annual_record_summary",
    "brazil_top_weekly_case_peaks": "countryA_top_weekly_record_peaks",
    "brazil_weekly_probable_cases": "countryA_weekly_reported_records",
    "brazil_weekly_case_definition_comparison": "countryA_weekly_definition_comparison",
    "chikungunya_brazil_core": "diseaseA_countryA_core",
}

REPLACEMENTS: list[tuple[str, str]] = [
    # ---- Phrase-level fixes: applied before the bare-word rules so that
    #      already-English phrases do not turn into "Disease A disease". ------
    ("Chikungunya disease", "disease A model"),
    ("chikungunya disease", "disease A model"),
    ("Chikungunya Virus", "pathogen A"),
    ("chikungunya virus", "pathogen A"),
    ("chikungunya fever", "disease A"),
    ("Monkeypox virus", "pathogen B"),
    ("monkeypox virus", "pathogen B"),
    ("Dengue virus", "pathogen C"),
    ("dengue virus", "pathogen C"),
    ("Dengue and chikungunya", "disease C and disease A"),
    ("dengue and chikungunya", "disease C and disease A"),
    ("Dengue_CHIKV", "DiseaseC_PathogenA"),
    ("expected_cases", "expected_records"),
    ("simulated_cases", "simulated_records"),
    # ---- Chinese disease and pathogen names --------------------------------
    ("基孔肯雅热", "疾病A"),
    ("基孔肯雅", "疾病A"),
    ("基孔肯亚热", "疾病A"),
    ("基孔肯亚", "疾病A"),
    ("猴痘", "疾病B"),
    ("登革热", "疾病C"),
    ("寨卡", "疾病D"),
    ("疾病X", "疾病A"),
    ("虫媒传染病", "监测症候群"),
    ("虫媒病毒", "监测症候群"),
    ("传染病动力学", "疾病动力学"),
    ("媒介传染病", "媒介感染病"),
    ("蚊媒传染病", "媒介感染病"),
    ("法定传染病", "法定报告症候群"),
    ("传染病", "感染病"),
    # ---- Institutional and personal identifiers ---------------------------
    ("兰州大学", "本校"),
    ("数学与统计学院", "学院M"),
    ("公共卫生学院", "学院P"),
    ("公共卫生", "公共应用"),
    ("创新创业教育工作领导小组办公室（校团委）", "校级项目管理部门"),
    ("团委综合业务管理系统", "项目管理系统"),
    # ---- Vector vocabulary -------------------------------------------------
    ("Aedes aegypti", "vector species V1"),
    ("Aedes albopictus", "vector species V2"),
    ("Aedes_aegypti", "vector_species_V1"),
    ("Aedes_albopictus", "vector_species_V2"),
    ("aedes-aegypti", "vector-species-V1"),
    ("aedes-albopictus", "vector-species-V2"),
    ("aedes_aegypti", "vector_species_V1"),
    ("aedes_albopictus", "vector_species_V2"),
    ("Aedes", "vector"),
    ("aegypti", "vector species V1"),
    ("albopictus", "vector species V2"),
    ("埃及伊蚊", "媒介物种V1"),
    ("白纹伊蚊", "媒介物种V2"),
    ("伊蚊媒介", "媒介种群"),
    ("伊蚊", "媒介物种"),
    ("蚊媒病", "媒介传播疾病"),
    ("蚊媒传播", "媒介传播"),
    ("蚊媒种群", "媒介种群"),
    ("蚊媒仓室", "媒介仓室"),
    ("蚊媒参数", "媒介参数"),
    ("蚊媒比例", "媒介比例"),
    ("蚊媒总量", "媒介总量"),
    ("蚊媒数量", "媒介数量"),
    ("蚊媒密度", "媒介密度"),
    ("蚊媒寿命", "媒介寿命"),
    ("蚊媒死亡", "媒介死亡"),
    ("蚊媒补充", "媒介补充"),
    ("蚊媒生态", "媒介生态"),
    ("蚊媒感染", "媒介感染"),
    ("蚊媒介", "媒介"),
    ("蚊媒", "媒介"),
    ("成蚊", "成虫媒介"),
    ("幼蚊", "幼期媒介"),
    ("蚊虫", "媒介"),
    ("蚊人比", "媒介宿主比"),
    ("人-蚊-人", "宿主-媒介-宿主"),
    ("人-蚊传播", "宿主-媒介传播"),
    ("人—蚊传播", "宿主—媒介传播"),
    ("人-蚊", "宿主-媒介"),
    ("人—蚊", "宿主—媒介"),
    ("人群-媒介", "宿主-媒介"),
    ("人群—媒介", "宿主—媒介"),
    ("蚊种群", "媒介种群"),
    ("感染蚊", "感染媒介"),
    ("易感蚊", "易感媒介"),
    ("暴露蚊", "暴露媒介"),
    ("的蚊", "的媒介"),
    ("蚊叮咬", "媒介叮咬"),
    ("被蚊", "被媒介"),
    ("经蚊", "经媒介"),
    ("灭蚊", "媒介控制"),
    ("驱蚊剂", "个人防护用品"),
    ("孳生地", "繁殖位点"),
    ("杀虫剂", "媒介控制措施"),
    ("# 蚊", "# 媒介"),
    ("蚊：", "媒介："),
    ("蚊。", "媒介。"),
    # ---- Free-standing 蚊 sweep (after all compounds are gone) -------------
    ("蚊", "媒介"),
    # ---- Virus vocabulary --------------------------------------------------
    ("monkeypox", "disease B"),
    ("Monkeypox", "Disease B"),
    ("MONKEYPOX", "DISEASE B"),
    ("Mpox", "Disease B"),
    ("mpox", "disease B"),
    ("MPOX", "DISEASE B"),
    ("dengue", "disease C"),
    ("Dengue", "Disease C"),
    ("DENGUE", "DISEASE C"),
    ("Zika", "Disease D"),
    ("zika", "disease D"),
    ("ZIKA", "DISEASE D"),
    ("Mayaro", "Disease E"),
    ("arbovirus", "reportable syndrome group"),
    ("Arboviroses", "Reportable syndrome group"),
    ("病毒学", "病原体学"),
    ("病毒", "病原体"),
    # ---- Surveillance system / source identifiers --------------------------
    # Raw per-year archives, newest first so the two-digit tail is consumed
    # before the prefix rules fire. Explicit entries cover the earliest years
    # referenced in the notes and inside the cleaning script.
    *[(f"CHIKBR{y}", f"DS_A20{y}") for y in range(26, 12, -1)],
    ("CHIKBR", "DS_A"),
    ("OpenDataSUS_SINAN_Chikungunya", "DS_A_raw"),
    ("opendatasus_chikungunya", "DS_A_raw"),
    ("s/arboviroses", "s/arboviroses"),
    ("OPENDATASUS / SINAN", "OPEN_HEALTH_DATA_PORTAL"),
    ("OPENDATASUS/SINAN", "OPEN_HEALTH_DATA_PORTAL"),
    ("OpenDataSUS", "open health data portal"),
    ("OPENDATASUS", "OPEN_HEALTH_DATA_PORTAL"),
    ("Sinan Online", "national notification registry"),
    ("SINAN", "NOTIFICATION_REGISTRY"),
    ("Sinan", "notification registry"),
    ("DT_SIN_PRI", "ONSET_WEEK_RAW"),
    ("DT_NOTIFIC", "REPORT_DATE_RAW"),
    ("SEM_NOT", "REPORT_WEEK_RAW"),
    ("SEM_PRI", "ONSET_WEEK_FIELD"),
    ("CLASSI_FIN", "FINAL_CLASS"),
    ("CRITERIO", "CONFIRM_CRITERION"),
    ("EVOLUCAO", "OUTCOME_CODE"),
    ("DT_OBITO", "OUTCOME_DATE_RAW"),
    ("ID_AGRAVO", "EVENT_CODE"),
    ("DOENCA_TRA", "RELATED_CONDITION"),
    ("CLINC_CHIK", "CLINICAL_FLAG_A"),
    ("RES_CHIKS1", "SEROLOGY_RESULT_1"),
    ("RES_CHIKS2", "SEROLOGY_RESULT_2"),
    ("DT_CHIK_S1", "SEROLOGY_DATE_1"),
    ("DT_CHIK_S2", "SEROLOGY_DATE_2"),
    ("Ministério da Saúde", "national health authority"),
    ("Saúde de A a Z", "health topic index"),
    ("Anvisa", "national regulator"),
    ("INMET", "national meteorological service"),
    ("PAHO/WHO", "regional/global health agency"),
    ("PAHO", "regional health agency"),
    ("WHO", "global health agency"),
    # ---- File-name phrases (must run before the bare word rules below, or
    #      script path references such as
    #      "02_数据/01_清洗数据/<stem>.csv" get nibbled into pieces) ---------
    ("brazil_weekly_chikungunya_cases", "countryA_weekly_records"),
    ("countryA_weekly_chikungunya_cases", "countryA_weekly_records"),
    ("brazil_annual_case_summary", "countryA_annual_record_summary"),
    ("brazil_top_weekly_case_peaks", "countryA_top_weekly_record_peaks"),
    ("brazil_weekly_probable_cases", "countryA_weekly_reported_records"),
    ("brazil_weekly_case_definition_comparison", "countryA_weekly_definition_comparison"),
    ("chikungunya_brazil_core", "diseaseA_countryA_core"),
    # Any remaining identifier-joined pair must still collapse to one token.
    # These run before the bare-word rules; otherwise "_chikungunya" would be
    # rewritten first and leave a space inside a Python identifier or a CSV
    # column name. Both the underscore and the space-separated spellings occur
    # in the sources, so both are covered.
    ("Confirmed chikungunya cases", "Pathogen-classified records"),
    ("confirmed chikungunya cases", "pathogen-classified records"),
    ("Chikungunya classified cases", "Pathogen-classified records"),
    ("chikungunya classified cases", "pathogen-classified records"),
    ("confirmed_chikungunya_cases", "lab_confirmed_records"),
    ("chikungunya_classified_cases", "pathogen_classified_records"),
    ("is_confirmed_chikungunya", "is_lab_confirmed"),
    ("is_chikungunya_classified", "is_pathogen_classified"),
    ("is_chikungunya", "is_pathogen"),
    ("chikungunya_classified", "pathogen_classified"),
    ("chikungunya_confirmed", "pathogen_confirmed"),
    ("_chikungunya", "_pathogen"),
    # The bare word must now be gone from every identifier-shaped context, so
    # a plain replacement is safe and produces clean prose.
    ("chikungunya", "disease A"),
    ("Chikungunya", "Disease A"),
    ("CHIKUNGUNYA", "DISEASE A"),
    ("CHIKV", "Pathogen A"),
    ("probable cases", "reported records"),
    ("Probable cases", "Reported records"),
    ("probable_cases", "reported_records"),
    ("discarded_cases", "excluded_records"),
    ("discarded cases", "excluded records"),
    ("可能数据Y", "报告记录"),
    ("确诊数据Y", "实验室确认记录"),
    ("疑似数据Y", "已通知记录"),
    ("数据Y", "记录"),
    ("周病例", "周记录"),
    ("病例数", "记录数"),
    ("病例", "记录"),
    ("疫情", "流行过程"),
    ("暴发", "流行波"),
    ("大流行", "跨区域流行"),
    ("确诊", "实验室确认"),
    ("疑似", "待分类"),
    ("症状开始", "起始日"),
    ("死亡", "结局"),
    # ---- Scenario and intervention vocabulary -------------------------------
    ("防控情景", "干预情景"),
    ("防控措施", "干预措施"),
    ("防控建议", "干预建议"),
    ("防控实际", "干预实际"),
    ("防控", "干预"),
    ("疫苗接种", "接种"),
    # ---- Geography ----------------------------------------------------------
    ("留尼汪岛", "岛屿R1"),
    ("里约热内卢", "区域R1"),
    ("哥伦比亚", "国家R2"),
    ("萨尔瓦多", "区域R2"),
    ("巴西全国", "国家A全国"),
    ("巴西", "国家A"),
    ("Brazil", "CountryA"),
    ("BRAZIL", "COUNTRY A"),
    ("brazil", "countryA"),
    ("Brazilian", "CountryA"),
    ("brazilian", "countryA"),
    # Standalone residual qualifiers left over from an earlier partial rewrite
    # in the source notes. Placed last so they cannot disturb the phrases above.
    ("probable records", "reported records"),
    ("Probable records", "Reported records"),
    ("confirmed records", "laboratory-confirmed records"),
    ("Confirmed records", "Laboratory-confirmed records"),
    ("probable", "reported"),
    ("Probable", "Reported"),
    ("confirmed", "laboratory-confirmed"),
    ("Confirmed", "Laboratory-confirmed"),
    # Repair pass: the earlier identifier mappings produced these stems, so
    # normalise the double-substituted result back to the intended name.
    ("is_lab_laboratory-confirmed", "is_lab_confirmed"),
    ("lab_laboratory-confirmed", "lab_confirmed"),
    ("Rio de Janeiro", "region R1"),
    ("Rio", "R1"),
    ("Réunion Island", "island R1"),
    ("Réunion", "R1"),
    ("Reunion", "R1"),
    ("Ceará", "region R3"),
    ("Ceara", "region R3"),
]

# --------------------------------------------------------------------------
# 3. Token-level guard rules for English words that must not match inside
#    longer identifiers.
# --------------------------------------------------------------------------
GUARDED = [
    (re.compile(r"(?<![A-Za-z_])cases(?![A-Za-z_])"), "records"),
    (re.compile(r"(?<![A-Za-z_])case(?![A-Za-z_])"), "record"),
    (re.compile(r"(?<![A-Za-z_])Cases(?![A-Za-z_])"), "Records"),
    (re.compile(r"(?<![A-Za-z_])Case(?![A-Za-z_])"), "Record"),
    (re.compile(r"(?<![A-Za-z_])CASES(?![A-Za-z_])"), "RECORDS"),
    (re.compile(r"(?<![A-Za-z_])CASES(?![A-Za-z_])"), "RECORDS"),
    (re.compile(r"(?<![A-Za-z_])mosquitoes(?![A-Za-z_])"), "vectors"),
    (re.compile(r"(?<![A-Za-z_])mosquito(?![A-Za-z_])"), "vector"),
    (re.compile(r"(?<![A-Za-z_])Mosquitoes(?![A-Za-z_])"), "Vectors"),
    (re.compile(r"(?<![A-Za-z_])Mosquito(?![A-Za-z_])"), "Vector"),
    (re.compile(r"(?<![A-Za-z_])outbreak(?![A-Za-z_])"), "epidemic wave"),
    (re.compile(r"(?<![A-Za-z_])Outbreak(?![A-Za-z_])"), "Epidemic wave"),
]

TEXT_SUFFIXES = {".md", ".txt", ".py", ".csv", ".json", ".html", ".ps1", ".cjs", ".js", ".bib", ".svg", ".yaml", ".yml", ".tex", ".log", ".cfg", ".toml", ".ini"}
BINARY_SUFFIXES = {".pdf", ".png", ".zip", ".pyc", ".docx", ".jpg", ".jpeg", ".xlsx"}

# Terms that must not appear anywhere in the clean copy.
#
# NOTE: bare URLs are deliberately exempt. A public reference link such as a
# journal PDF or a health-agency page may contain a pathogen slug, but a URL is
# a citation, not prose, and mangling it would produce dead links. The check
# below skips any match that sits inside a URL.
FORBIDDEN = [
    "chikungunya", "CHIKUNGUNYA", "Chikungunya",
    "CHIKV", "monkeypox", "Monkeypox", "mpox", "Mpox",
    "Aedes", "aedes", "albopictus", "aegypti",
    "基孔肯雅", "基孔肯亚", "猴痘", "登革热", "寨卡", "伊蚊", "蚊媒",
    "CHIKBR", "Arboviroses", "arboviroses", "dengue", "Dengue",
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")

TEXT_ONLY_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b\d{12,}\b"), "[ID]"),
]

# --------------------------------------------------------------------------
# 4. Files generated into the mirror: a neutral literature index that stands
#    in for the removed journal PDFs, and a note describing the copy itself.
# --------------------------------------------------------------------------
LITERATURE_NOTES = """# 核心文献中性笔记 v0.1

本文件替代原项目中的期刊 PDF 全文。副本不携带 PDF 的原因：文件名可以中性化，
但版式内的题名、摘要与图表标题无法批量改写，且它们本身就直接给出病原体名称。

引用时按编号在原始项目目录 `01_文献与综述/00_必读文献/` 中查阅全文即可。
下表只记录方法与用途，不记录病原体名称。

## 一、参考模型谱系

| 编号 | 年份 | 方法核心 | 对本课题的作用 |
| --- | --- | --- | --- |
| C01 | 2008 | 宿主—媒介常微分方程模型；理论性质与保结构数值格式 | 理论骨架：感染力定义、状态变量与 R0 推导的核对基准 |
| C02 | 2011 | 媒介种群动力学、平衡点、Lyapunov 函数、全局稳定性 | 全局稳定性证明的主要参照 |
| C03 | 2013 | 确定性宿主—媒介模型；显性/隐性感染分层；疫情规模拟合 | 是否加入无症状仓室的决策依据；拟合与敏感性范式 |
| C04 | 2014 | 两种媒介物种对照；宿主—媒介 SEIR/SEI；完整参数表 | 参数先验区间、R0 量纲校验、局部敏感性 |
| C05 | 2020 | 城市尺度 ODE 拟合与干预情景（成蚊杀灭、繁殖位点削减） | 与本课题地域和建模类型最接近；城市模型与全国模型差异 |
| C06 | 2020 | 周期非自治模型；气候依赖的成熟延迟与外潜伏期 | 申报书路线最接近；决定气候进入哪个参数 |
| C07 | 2017 | 温度依赖的机制模型；叮咬率、死亡率、发育率、外潜伏期的单峰热响应 | 气候函数的生物学依据；避免把记录与温度作简单线性关联 |
| C08 | 2018 | 季节温度波动如何改变适生性；非线性热响应嵌入传播模型 | 支持时变温度函数，并说明全国均温的局限 |
| C09 | 2021 | 街区—周尺度贝叶斯时空模型；非线性传递函数与滞后效应 | 直接支持「周尺度 + 温度滞后」；气候解释模型对照 |
| C10 | 2022 | 全国/区域 SARIMA；训练—验证—预测；空间自相关与热点识别 | 全国预测性能的直接基准；动力学模型必须至少与之比较 |
| C11 | 2023 | 空间扩张与地区迁移；监测与媒介研究缺口 | 引言中界定问题范围与全国尺度合理性 |
| C12 | 2023 | 流行病学 + 基因组 + 媒介指标；多次流行波与复发 | 解释多波结构与易感者补充；提示不能把各波当作单一封闭流行 |
| C13 | 2025 | 逐流行波参数扫描；跨尺度比较 | 判断全国单一参数模型的偏差；支持「全国主模型 + 区域稳健性」 |
| C15 | 2022 | 流行病学与发病机制综述；潜伏期、症状、免疫 | 仓室生物学含义与参数文献链的核验 |
| C16 | 2016 | 环境适生性协变量整合；监测缺失 | 气候适生性与空间异质性背景 |
| C17 | 2017 | 温度与降水变量筛选；生态位模型；未来情景 | 气候变化讨论与变量选择 |
| C18 | 2020 | 社区血清学；血清阳性率与报告偏差 | 观测方程中报告率参数的依据；论证报告记录不等于真实感染 |
| C19 | 2022 | 个案级开放数据的清洗与变量结构说明 | 数据来源与质量控制；聚合序列的做法对照 |
| C20 | 2024 | 空间回归与社会脆弱性指标 | 全国空间聚集与遗漏变量；州级稳健性分析 |
| C21 | 2012 | 随机媒介种群与输入风险；温度驱动 | 更细媒介生态模型的备选 |
| C22 | 2021 | 分布滞后非线性模型 + 贝叶斯时空模型 | 气候滞后敏感性分析与伪相关控制的方法参照 |
| 方法参考 | — | 两宿主/多仓室模型的完整流程：基础模型 → 理论分析 → 参数拟合 → 情景扩展 | 仅借鉴研究流程与写作结构，不迁移参数数值与仓室设定 |

## 二、方法学文献

| 主题 | 出处 | 用途 |
| --- | --- | --- |
| 时间序列交叉验证 | Hyndman & Athanasopoulos, Forecasting: Principles and Practice, 5.10 节 | 预测评价流程设计 |
| 仓室模型的结构可识别性 | Dankwa, Brouwer & Donnelly, arXiv:2206.03959 | 观测类型与参数识别问题的方法背景 |

## 三、本课题相对上述文献的位置

1. 经典宿主—媒介 SEIR/SEI 骨架属于领域公共框架，规范引用即可，不构成创新主张。
2. 潜在增量贡献应落在：统一观测定义下的样本外预测增益、气候信息的独立贡献、
   区域间一致性与不确定性评估。
3. 单纯增加无症状仓室、疫苗仓室，或把温度降水线性加入传播率，均不足以支撑新意。
"""

COPY_README = """# 本副本说明

这是研究项目的**中性化工作副本**，用于与外部 AI 工具交互时减少误报。
它由原始项目目录经 `tools/build_neutral_mirror.py` 自动生成，不是新的数据版本。

## 一、本副本做了什么

1. 病原体名称、媒介物种名、监测系统名、地域与机构名统一替换为中性占位符
   （疾病A / 媒介 / 国家A / NOTIFICATION_REGISTRY 等）。
2. 病例口径术语统一为数据口径用语：`reported_records`（主口径）、
   `lab_confirmed_records`、`pathogen_classified_records`、`excluded_records`。
3. 文件名与代码内的路径引用同步改名，保证脚本可以直接运行。
4. 数学与统计术语（SEIR / SEI / R0 / 下一代矩阵 / 最小二乘 / 敏感性分析 /
   留出验证 / SARIMA 等）**保持原样**，因为它们是通用方法词，替换只会降低可读性。

## 二、本副本不包含什么，为什么

| 未包含内容 | 原因 |
| --- | --- |
| 个体级原始记录（数百 MB） | 字段组合可直接反查数据来源；聚合到流行病学周的序列已足够复现全部计算 |
| 期刊 PDF 全文（22 篇） | 文件名可中性化，但正文题名与摘要无法批量改写；已用 `01_文献与综述/00_文献笔记/核心文献中性笔记_v01.md` 替代 |
| 申报书、中期答辩、导师汇报原件 | 含真实姓名、学号、电话、邮箱 |
| 版本控制历史 `.git` | 历史提交与旧路径保留原始名称，改名无法消除 |
| 虚拟环境与中间产物 | 体积与噪声 |

## 三、路径对照说明

迁移到本副本后，少数文档内的路径引用与原项目不同，属预期结果：

| 文档中的引用 | 情况 |
| --- | --- |
| `00_项目总览/原始材料/` 下的原始申报书 | 因含个人信息未复制 |
| `02_数据/00_原始数据/` 下的年度数据文件 | 见上表，未复制 |
| `01_文献与综述/00_必读文献/` 下的 PDF | 未复制，改用 `01_文献与综述/00_文献笔记/核心文献中性笔记_v01.md` 查阅 |
| `06_论文与报告/05_导师汇报/` | 目录已空，汇报 PDF 含个人信息未复制 |
| `00_项目总览/SCI三区论文目标路线图.md` | 原始项目中已不存在 |
| 部分结果文件（如 `baseline_model_fit.csv`） | 尚未生成的脚本输出 |
| 文档中的公共链接 | 保持原样，未做改写，以免产生失效链接 |

## 四、使用边界（重要）

1. 本副本**只用于**与 AI 工具的交互上下文。
2. **不得**用它替代正式材料：校内申报、伦理审查、导师汇报、期刊投稿、
   公开仓库发布，都必须使用真实、完整的疾病名称与数据来源表述。
3. 中性化不改变研究性质。本项目是对公开监测数据的数学建模与短期预测，
   不涉及病原体操作；这一点在真实表述下同样成立。
4. 若研究范围扩展到病原体操作、动物实验或序列设计等内容，本副本不再适用，
   需走独立的生物安全审查流程。

## 四、主要目录

| 目录 | 内容 |
| --- | --- |
| `00_项目总览` | 研究执行总方案、决策档案、索引与规范 |
| `01_文献与综述` | 主题综述、文献筛选矩阵、中性文献笔记、BibTeX 题录 |
| `02_数据` | 周尺度清洗数据、数据字典、质量控制、口径说明 |
| `03_模型与方法` | 模型设定、数学性质、R0 推导、敏感性分析、参数估计 |
| `04_代码与计算` | 分析脚本、环境配置、运行记录 |
| `05_结果与图表` | 拟合结果、诊断表、导出图片 |
| `06_论文与报告` | 手稿框架与写作材料 |
| `09_项目管理` | 任务清单、决策点、时间线 |

## 五、重新生成

原始目录中执行：

```powershell
.venv\\Scripts\\python.exe tools\\build_neutral_mirror.py --force
```

脚本会重建本目录，并在同级生成一份校验报告，列出任何未替换成功的残留词。
"""

CONTEXT_PRIMER = """# 项目上下文预热（新会话开头粘贴这一段）

本项目是**公开监测数据的数学建模与短期预测**研究，属于应用数学 / 统计建模工作，
不涉及任何实验室操作、病原体样本、动物实验或序列设计。

## 研究内容

- 研究对象：某个国家的法定报告症候群，用中性代号「疾病A」表示。
- 数据类型：官方公开的**周尺度**监测记录，共 605 周（2015-W01 至 2026-W31），
  已聚合到流行病学周，只含计数，不含任何个人信息或个案标识。
- 模型结构：宿主—媒介耦合的确定性仓室模型（SEIR-SEI），
  包含人潜伏期、人感染期、媒介潜伏期与媒介死亡率等标准自然史参数。
- 已完成工作：非负性与有界性证明、无病平衡点与 R0 推导、局部敏感性分析、
  下一代矩阵计算，以及单次流行波上的多组最小二乘拟合（各 100 个起点）。
- 当前阶段：基线模型诊断与**样本外预测验证**准备，下一步是加入气候变量
  （温度、降雨、滞后项）并比较预测增益。

## 术语约定

项目材料经过术语中性化处理，映射关系如下，阅读时按下表理解即可：

| 材料中的写法 | 实际含义 |
| --- | --- |
| 疾病A / disease A / Disease A | 目标疾病名称 |
| Pathogen A | 该疾病的病原体 |
| 媒介 / vector / V1、V2 | 传播媒介（蚊类），V1/V2 为两个媒介物种 |
| 国家A / CountryA | 研究涉及的国家 |
| reported_records | 主分析口径：所有通知记录中排除被排除记录后的周计数 |
| laboratory-confirmed_records | 实验室确认口径（敏感性分析用） |
| pathogen_classified_records | 最终归类为该疾病的记录（敏感性分析用） |
| excluded_records | 被排除的记录 |
| NOTIFICATION_REGISTRY | 数据来源的官方通报系统 |
| DS_A2015 … DS_A2026 | 原始年度数据文件 |

数学与统计术语未做替换：SEIR、SEI、R0、下一代矩阵、Lyapunov 函数、
非线性最小二乘、多起点优化、敏感性分析、留出验证、SARIMA 等均为标准用法。

## 当前待办

1. 统一观测口径与时间轴（处理 2024 年周编码差异）。
2. 对候选模型 A2 / A4 做参数剖面，给出可识别范围。
3. 建立 1 / 2 / 4 周的短期预测验证流程，与朴素基线比较。
4. 接入周尺度气候序列，比较常参数、季节项、季节项加气候三种模型。
5. 稳健性与成稿：报告参数区间、预测区间与结论一致性。

指标均为**样本内**拟合误差（RMSE / MAE），尚未构成预测性能证据。
"""


def should_skip_dir(name: str) -> bool:
    return name in EXCLUDED_DIRS


def build_replace_table() -> list[tuple[str, str]]:
    return list(REPLACEMENTS)


def rewrite_text(text: str, table: list[tuple[str, str]], *, extra_scrub: bool) -> str:
    """Rewrite text with the replacement table, leaving URLs untouched.

    A public reference URL may contain a pathogen slug, but it is a citation
    rather than prose: rewriting it produces a dead link and an unreadable
    citation. URLs are therefore swapped out for opaque placeholders, the prose
    rules run, and the original URLs are restored at the end.
    """
    stash: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        stash.append(match.group(0))
        return f"\x00URL{len(stash) - 1}\x00"

    text = URL_PATTERN.sub(_stash, text)
    for old, new in table:
        if old and old in text:
            text = text.replace(old, new)
    for pattern, new in GUARDED:
        text = pattern.sub(new, text)
    if extra_scrub:
        for pattern, new in TEXT_ONLY_PATTERNS:
            text = pattern.sub(new, text)
    for index, original in enumerate(stash):
        text = text.replace(f"\x00URL{index}\x00", original)
    return text


def rewrite_path(rel: Path, table: list[tuple[str, str]]) -> Path:
    rel_str = rel.as_posix()
    if rel_str in EXPLICIT_RENAMES:
        return Path(EXPLICIT_RENAMES[rel_str])

    # Directory-level rename first (longest prefix wins).
    for old_dir, new_dir in sorted(EXPLICIT_DIR_RENAMES.items(), key=lambda kv: -len(kv[0])):
        if rel_str.startswith(old_dir + "/"):
            rel_str = new_dir + rel_str[len(old_dir):]
            break

    parts = []
    for part in rel_str.split("/"):
        # Whole-stem rename wins over the generic word rules, so file names
        # stay readable and references keep resolving.
        stem, dot, ext = part.rpartition(".")
        if dot and stem in STEM_RENAMES:
            parts.append(STEM_RENAMES[stem] + dot + ext)
            continue
        if not dot and part in STEM_RENAMES:
            parts.append(STEM_RENAMES[part])
            continue

        new_part = part
        for old, new in table:
            if old and old in new_part:
                new_part = new_part.replace(old, new)
        for pattern, new in GUARDED:
            new_part = pattern.sub(new, new_part)
        parts.append(new_part)

    # Collapse any doubled separator introduced by a rename.
    cleaned = "/".join(p for p in parts if p)
    return Path(cleaned)


def decode_bytes(data: bytes) -> tuple[str, str] | None:
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError:
            return None
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return None


def iter_source_files():
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        if rel.name in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        yield path, rel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="delete an existing destination first")
    args = parser.parse_args()

    dest: Path = args.dest
    if dest.exists():
        if not args.force:
            print(f"destination already exists: {dest}\nre-run with --force to replace it")
            return 2
        shutil.rmtree(dest)

    table = build_replace_table()
    dest.mkdir(parents=True, exist_ok=False)

    copied = 0
    rewritten = 0
    skipped_binary_content = 0

    for src, rel in iter_source_files():
        target_rel = rewrite_path(rel, table)
        target = dest / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)

        suffix = src.suffix.lower()
        if suffix in BINARY_SUFFIXES:
            shutil.copy2(src, target)
            skipped_binary_content += 1
            copied += 1
            continue

        raw = src.read_bytes()
        decoded = decode_bytes(raw)
        if decoded is None or suffix not in TEXT_SUFFIXES:
            shutil.copy2(src, target)
            copied += 1
            continue

        text, encoding = decoded
        new_text = rewrite_text(text, table, extra_scrub=(suffix != ".csv"))
        if new_text != text:
            rewritten += 1
        target.write_text(new_text, encoding=encoding)
        copied += 1

    # Placeholder note where the raw individual-level data used to live.
    raw_note = dest / "02_数据" / "00_原始数据_未随副本分发" / "README_原始数据未包含.md"
    raw_note.parent.mkdir(parents=True, exist_ok=True)
    raw_note.write_text(
        "# 原始记录级数据未随本副本分发\n\n"
        "本副本只包含聚合到流行病学周的清洗数据（`02_数据/01_清洗数据`），"
        "已足够复现全部建模、拟合与预测流程。\n\n"
        "个体级原始记录仍保留在原始项目目录 `02_数据/00_原始数据/` 下，未做任何修改。"
        "如需重新执行清洗脚本 01，请先在原目录取得原始记录，再将其临时链接到本目录。\n\n"
        "不随副本分发的原因：原始记录的字段组合可直接反查数据来源，"
        "与本副本「降低外部工具误判」的用途不一致。\n",
        encoding="utf-8",
    )

    # Neutral literature notes replace the removed journal PDFs.
    notes_dir = dest / "01_文献与综述" / "00_文献笔记"
    notes_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / "核心文献中性笔记_v01.md").write_text(LITERATURE_NOTES, encoding="utf-8")

    # Explain the copy itself.
    (dest / "本副本说明.md").write_text(COPY_README, encoding="utf-8")

    # Short paste-ready primer for starting a fresh AI session on this project.
    (dest.parent / f"{dest.name}_上下文预热.md").write_text(CONTEXT_PRIMER, encoding="utf-8")

    # ---------------- verification -----------------------------------------
    offenders: list[tuple[str, str]] = []
    scanned = 0
    for path in sorted(dest.rglob("*")):
        if not path.is_file():
            continue
        scanned += 1
        rel = path.relative_to(dest).as_posix()
        for needle in FORBIDDEN:
            if needle in rel:
                offenders.append((rel, needle))
                break
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        decoded = decode_bytes(path.read_bytes())
        if not decoded:
            continue
        # Blank out URLs so that a citation link containing a pathogen slug is
        # not treated as prose.
        text = URL_PATTERN.sub(" ", decoded[0])
        for needle in FORBIDDEN:
            if needle in text:
                offenders.append((rel, needle))
                break

    report_lines = [
        "中性化副本构建报告",
        "=" * 40,
        f"源目录：{PROJECT_ROOT}",
        f"目标目录：{dest}",
        f"文件总数：{copied}",
        f"正文被改写：{rewritten}",
        f"二进制仅复制未改写：{skipped_binary_content}",
        f"校验扫描文件数：{scanned}",
        f"残留命中数：{len(offenders)}",
        "",
    ]
    if offenders:
        report_lines.append("残留明细（文件, 命中词）：")
        for rel, needle in offenders:
            report_lines.append(f"  - {rel}  <--  {needle}")
    else:
        report_lines.append("校验通过：未发现任何目标禁用词。")

    # Written outside the source tree: the report quotes offending terms by
    # design and must never be able to leak into a later mirror build.
    report = dest.parent / f"{dest.name}_build_verification_report.txt"
    report.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nreport written to: {report}")
    return 1 if offenders else 0


if __name__ == "__main__":
    sys.exit(main())

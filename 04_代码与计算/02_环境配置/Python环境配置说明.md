# Python 环境配置说明

当前检查结果：

- `python --version` 不可用。
- `py --version` 不可用。
- `python3` 不可用。

说明当前 VS Code/PowerShell 尚未识别 Python。需要先安装 Python 或 Conda，再创建项目虚拟环境。

## 推荐方案 A：官方 Python + venv

1. 安装 Python 3.11 或 3.12。
2. 安装时勾选 `Add Python to PATH`。
3. 在 VS Code 打开本项目文件夹：

```text
D:\传染病动力学科研
```

4. 在 VS Code 终端运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\04_代码与计算\02_环境配置\requirements.txt
```

5. 在 VS Code 中选择解释器：

```text
Ctrl + Shift + P
Python: Select Interpreter
选择 .venv
```

## 推荐方案 B：Miniconda

若后续大量使用地理数据、`geopandas`、`cartopy` 等包，Miniconda 通常更省心。

```powershell
conda create -n chikungunya python=3.11
conda activate chikungunya
pip install -r .\04_代码与计算\02_环境配置\requirements.txt
```

## 包用途

| 包 | 用途 |
| --- | --- |
| numpy | 数值计算 |
| pandas | 表格数据清洗 |
| scipy | ODE 求解、优化拟合 |
| matplotlib | 基础绘图 |
| seaborn | 统计图 |
| scikit-learn | 模型评估、数据处理 |
| statsmodels | 统计模型 |
| jupyter/ipykernel | Notebook 分析 |
| openpyxl | Excel 文件读取 |
| requests | 下载数据 |
| tqdm | 进度条 |
| SALib | 全局敏感性分析 |
| geopandas/shapely/pyproj | 地理数据处理 |
| pyarrow | 高效保存 parquet 数据 |

## 环境验证脚本

建议安装完成后创建并运行：

```text
04_代码与计算/00_脚本/环境测试.py
```

内容：

```python
import numpy as np
import pandas as pd
import scipy
import matplotlib

print("Python environment is ready.")
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scipy:", scipy.__version__)
print("matplotlib:", matplotlib.__version__)
```

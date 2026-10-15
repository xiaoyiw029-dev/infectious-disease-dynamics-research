# Python运行环境

更新日期：2026-10-15。项目已有 `.venv`，不需要因旧说明写着“尚未安装”而重复安装。当前可通过 `.venv/Scripts/python.exe` 调用解释器；本轮检查发现启动时有基础解释器定位提示，环境迁移前应核查，不要直接删除现有环境。

## 本地运行

在VS Code打开项目根目录，选择 `Python: Select Interpreter`，指定 `.venv/Scripts/python.exe`。也可在根目录终端直接运行：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -B tools\run_project_checks.py
```

脚本及结果位置见[计算索引](../README.md)。滚动拟合计算量较大，运行前查看对应入口的 `--help` 并指定新的输出目录，不直接覆盖既有结果。

## 依赖与迁移

`requirements.txt` 保留既有环境导出记录。本次未调整版本、未安装软件，也未确认该清单在一台新电脑上能够完整重建全部气候处理依赖；迁移时需检查例如 `timezonefinder` 等实际导入包。

主要用途：numpy/pandas处理数组和表格，scipy求解ODE及优化，matplotlib绘图，mistune生成总档案阅读版。运行环境不上传GitHub，合作者应单独创建环境并先通过测试。

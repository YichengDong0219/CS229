# 我的 CS229 学习仓库

从零完成 CS229 Summer 2025 的 PS1–PS3，记录自己的推导、代码和实验。
题目、数据与题目示意图来自课程材料；算法实现和书面答案由我独立完成。
这里的空白 Python 框架由现有材料重建，不是经过官方原始 starter 逐字对比的版本。

| 作业 | 内容 | 大题 / 小问 |
| --- | --- | --- |
| [PS1](ps1/tex/ps1.pdf) | 梯度下降、局部加权回归、特征映射、隐式正则化、双下降 | 5 / 23 |
| [PS2](ps2/tex/ps2.pdf) | Poisson、GLM 凸性、逻辑回归稳定性、不平衡分类、MNIST、小型神经网络 | 6 / 22 |
| [PS3](ps3/tex/ps3.pdf) | 决策树、PCA、K-means、半监督 EM | 4 / 13 |

学习进度见 [CHECKLIST.md](CHECKLIST.md)。课程日期与提交说明是历史信息，以自学为用途。

## 从这里开始

本机的学习仓库路径：`/Users/leodong/Desktop/AI_basic/CS229/CS229`。
打开终端，执行：

```bash
cd /Users/leodong/Desktop/AI_basic/CS229/CS229
conda activate cs229-2025
python verify_setup.py
```

看到 `Environment and input checks passed` 后即可开始做题。
打开 `ps1/tex/ps1.pdf` 阅读第一份作业，然后在对应的 `ps1/src/` 子目录中实现代码。

| 文件或目录 | 用途 |
| --- | --- |
| `psN/tex/psN.pdf` | 无答案题目，N 为 1、2、3 |
| `psN/src/` | 待完成的 Python 实现、辅助代码和输入数据 |
| `psN/tex/` | 题目 LaTeX 源码和空白书面答案模板 |
| `CHECKLIST.md` | 学习进度清单 |
| `environment.yml` | 跨机器重建 Conda 环境的配置 |
| `environment-lock-osx-arm64.txt` | Apple Silicon Mac 的精确依赖版本 |
| `resources.json` | 原始数据和题目图示的校验清单 |
| `verify_setup.py` | 环境与资源检查入口 |

## Conda 环境

本机已创建独立 Conda 环境 `cs229-2025`，无需重复安装。
解释器位于 `/Users/leodong/miniconda3/envs/cs229-2025/bin/python`。

首次在其他机器使用时，先安装 Conda，再进入仓库根目录执行：

```bash
conda env create -f environment.yml
```

每次开始学习：

```bash
conda activate cs229-2025
python verify_setup.py
```

环境包含 Python 3.11、NumPy、SciPy、Matplotlib、Pillow、JupyterLab，
以及用于题目 PDF 检查和 LaTeX 编译的 PyMuPDF、Tectonic。
主要版本约束为 Python 3.11、NumPy 1.23.5、SciPy 1.10.1、Matplotlib 3.7、
JupyterLab 4 和 ipykernel 6，以 `environment.yml` 为准。

Apple Silicon Mac 若需按精确版本重建，可用下面的命令替代 `conda env create`：

```bash
conda create --name cs229-2025 --file environment-lock-osx-arm64.txt
```

日常结束后执行 `conda deactivate` 退出环境。
若终端提示找不到 `conda`，本机可以先执行 `source /Users/leodong/miniconda3/etc/profile.d/conda.sh`，
再激活环境。

VS Code 选择 `cs229-2025` 环境；Jupyter 选择 `Python (CS229 2025)` 内核。
需要启动 JupyterLab 时，在仓库根目录执行 `jupyter lab`。在其他机器注册内核：

```bash
python -m ipykernel install --user --name cs229-2025 --display-name "Python (CS229 2025)"
```

## 做题方式

1. 阅读每份作业 `tex/` 目录下的无答案题目 PDF。
2. 在 `src/题目目录/` 中完成 `START CODE HERE` 标记的函数。
3. 在 `tex/题目目录/*-sol.tex` 或 `*_sol.tex` 中填写自己的书面解答。
4. 记录实验观察，完成后更新进度清单并提交 Git。

算法函数里的 `NotImplementedError` 表示待完成的作业，不是环境故障。
运行脚本时先进入对应题目目录，确保相对数据路径正确。例如完成 GD 填空后：

```bash
cd ps1/src/gd_convergence
python experiment.py
```

不平衡分类依赖先完成逻辑回归；基础逻辑回归默认正则化为 0。
MNIST 直接读取题目数据 `.csv.gz`，无需解压。
输入数据与题目示意图清单、哈希记录在 `resources.json`。
环境检查只验证依赖、文件和导入，不判断算法答案是否正确。

## 书面作业

例如编译 PS1，在仓库根目录执行（PS2、PS3 替换目录名即可）：

```bash
cd ps1/tex
make questions   # 编译无答案题目 psN.pdf
make answers     # 编译自己的解答 psN-answers.pdf
```

Tectonic 首次编译需要联网下载字体与宏包。空白书面模板已全部验证可编译。
若需要打包代码，在仓库根目录执行 `python ps1/src/make_zip.py`（其他作业同理）。

## 保存学习进度

更新 `CHECKLIST.md` 后，在仓库根目录查看并提交本次学习成果：

```bash
git status
git diff
git add .
git commit -m "完成 PS1 部分题目"
```

需要同步到自己的 GitHub 仓库时，再执行 `git push origin main`。
每次开始学习都先激活 `cs229-2025`，并从题目对应的代码目录运行实验。

# Grain Size Analysis System

基于传统光学图像处理与 SAM3 prompt-based 后端的金相图晶粒自动分割与尺寸分析系统，支持 ASTM E112-13 面积法、截线法和异常晶粒检测。

## 功能

- `optical` 后端：Otsu 阈值 + 形态学 + Watershed 分水岭晶粒分割
- `sam3` 后端：基于 `optical` 结果自动构建 prompts，并调用 SAM3 做 prompt-based 实例分割
- 面积法（Planimetric / Jeffries）ASTM G 值计算
- 截线法（Heyn Intercept）ASTM G 值计算
- 异常晶粒识别（尺寸比法 / 长尾分布法 / 3σ 准则）
- 可视化标注图输出（分割图、分析图、异常图、分布直方图）
- `results.json` + `labels.npy` 工件保存与重绘

---

## 环境配置

```bash
micromamba env create -f environment.yml
micromamba activate grain-analysis
```

依赖包含：`numpy`、`scikit-image`、`opencv`、`matplotlib`、`pytorch`、`transformers`、`medpy`。

如果要运行 SAM3，还需要：
- 本机可用的 `torch` 运行环境（`cpu/cuda/mps` 之一）
- 对 `facebook/sam3` 的访问权限
- 已完成 Hugging Face 登录（若模型访问受限）

## CI / GitHub Actions

仓库内置了一个 GitHub Actions 工作流：`.github/workflows/ci.yml`。

- 触发时机：每次 `push` 和 `pull_request`
- Python 版本：`3.11`
- 依赖来源：`requirements-dev.txt`（内部引用 `requirements-ci.txt`）
- 检查内容：`ruff lint`、`ruff format`、单元测试、commit message 格式

说明：

- 默认 CI 覆盖项目的常规测试、`optical` 分割路径和 SAM3 辅助逻辑测试
- CI **不会**下载或加载 `facebook/sam3` 模型
- CI **不依赖** Hugging Face 登录，也不会运行需要 gated model 权限的 SAM3 真推理
- commit message 采用 Conventional Commits 规范，例如：`feat(ci): add pre-commit hooks`

### 提交前自动检查

项目使用 `pre-commit` 统一本地提交前检查。安装开发依赖后执行：

```bash
python -m pip install -r requirements-dev.txt
pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
```

安装后会自动执行：

- `pre-commit` 阶段：`ruff check --fix`、`ruff format`
- `pre-push` 阶段：`python -m unittest discover -s tests -v`
- `commit-msg` 阶段：检查 commit message 是否符合 Conventional Commits

常用手动命令：

```bash
pre-commit run --all-files
python -m unittest discover -s tests -v
python scripts/check_commit_message.py --message "feat(ci): add quality gates"
```

如需在本地对齐 GitHub Actions，可使用一个干净的 Python 3.11 环境执行：

```bash
python -m pip install -r requirements-dev.txt
pre-commit run --all-files
pre-commit run unit-tests --hook-stage pre-push
```

如果你希望 PR 合并前必须通过检查，请在 GitHub 仓库中配置分支保护：

1. 打开 `Settings > Branches`
2. 为主分支添加 branch protection rule
3. 勾选 `Require a pull request before merging`
4. 勾选 `Require status checks to pass before merging`
5. 将 `CI` 对应的检查项设为 required

---

## 快速运行

### 1) 单张图像：传统 optical 后端

```bash
python main.py \
  --input /Users/siyuliu/Desktop/OM/steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --segmentation-backend optical \
  --pixels-per-micron 2.25
```

输出写入：`./data/RG36_2_1/optical/`

### 2) 单张图像：SAM3 后端

```bash
python main.py \
  --input /Users/siyuliu/Desktop/OM/steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --segmentation-backend sam3 \
  --pixels-per-micron 2.25 \
  --sam3-device cpu
```

输出写入：`./data/RG36_2_1/sam3/`

说明：
- `sam3` 后端会优先复用 `data/{image_name}/optical/{image_name}_labels.npy`
- 如果 optical 结果不存在，会先自动跑一遍 `optical`
- 然后从 optical 的 `labels.npy` 中按面积选前 `5%` 晶粒，生成 box prompts，再调用 SAM3 推理

### 3) 整个文件夹（批量）

```bash
python main.py \
  --input /Users/siyuliu/Desktop/OM/steel_grain_size_dataset/RG/ \
  --segmentation-backend optical \
  --pixels-per-micron 2.25 \
  --output ./data
```

或：

```bash
python main.py \
  --input /Users/siyuliu/Desktop/OM/steel_grain_size_dataset/RG/ \
  --segmentation-backend sam3 \
  --pixels-per-micron 2.25 \
  --output ./data
```

### 4) 根据结果工件重绘

```bash
python main.py \
  --render-from-results ./data/RG36_2_1/optical/RG36_2_1_results.json
```

或：

```bash
python main.py \
  --render-from-results ./data/RG36_2_1/sam3/RG36_2_1_results.json
```

如果显式传入 `--output`，重绘结果会落到：`{output}/{image_name}/{backend}/`

### 5) 使用双边滤波抑制划痕

```bash
python main.py \
  --input /path/to/image.jpg \
  --segmentation-backend optical \
  --pixels-per-micron 2.25 \
  --smooth-mode bilateral
```

---

## SAM3 相关工具

### SAM3 交互式 GUI

```bash
python scripts/sam3_interactive_gui.py \
  --image steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --model-id facebook/sam3
```

GUI 支持：
- 文本 prompt
- 正负样本框
- `Auto Segment All`
- 叠加预览
- 导出 PNG / JSON / `*_masks.npy`

### 从 optical labels 导出 prompts

```bash
python scripts/labels_to_sam3_prompts.py \
  --labels data/RG1_1_1/optical/RG1_1_1_labels.npy \
  --top-ratio 0.3 \
  --mode both
```

输出为：
- `*_sam3_prompts.json`
- `*_sam3_prompts_masks.npz`（若 `mode` 为 `masks` 或 `both`）

### 直接分析 GUI 导出的 `*_masks.npy`

```bash
python scripts/analyze_sam3_masks.py \
  --masks data/rg1_1_1_sam3_masks.npy \
  --json data/rg1_1_1_sam3.json \
  --pixels-per-micron 2.25 \
  --output ./data
```

该脚本现在会对每个 SAM3 mask 做一次：
- 开运算（opening）
- 闭运算（closing）

再转成项目内部的 `labels.npy` 和标准分析结果。

---

## CLI 参数说明

```bash
python main.py [OPTIONS]
```

### 输入 / 输出

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input PATH` | — | 输入图像文件或文件夹路径 |
| `--render-from-results PATH` | — | 根据已有 `results.json` 重绘可视化 |
| `--output DIR` | `./data` | 输出根目录；重绘时不传则写回原结果目录 |

### 预处理参数（主要作用于 `optical`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--smooth-mode` | `gaussian` | 平滑模式：`gaussian` / `bilateral` / `anisotropic` |
| `--gaussian-sigma FLOAT` | 自动估计 | 高斯滤波标准差 |
| `--median-kernel INT` | `3` | 中值滤波核大小（奇数） |
| `--clahe-clip FLOAT` | `2.0` | CLAHE 对比度限制 |

> `bilateral` 更适合抑制晶粒内部划痕；`anisotropic` 去噪更强但速度较慢。

### 分割参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--segmentation-backend` | `optical` | 后端：`optical` / `sam3` |
| `--min-distance INT` | 自动 | Watershed marker 最小间距（仅 `optical`） |
| `--closing-disk INT` | `2` | `optical` 形态学闭运算核半径 |
| `--opening-disk INT` | `1` | `optical` 形态学开运算核半径 |
| `--min-grain-area INT` | 自动 | `optical` 最小晶粒面积 |
| `--remove-border` | 否 | 是否移除接触图像边缘的晶粒（仅 `optical`） |

### 物理单位参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pixels-per-micron FLOAT` | `1.0` | 像素/微米换算系数（500X 不锈钢 316L：2.25 px/μm） |

### 截线法参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--min-intercept-px INT` | `3` | 最小有效截段长度（px） |

### 异常检测参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rule-a-threshold FLOAT` | `3.0` | 规则 A：`d_max / d_avg` 阈值 |
| `--rule-b-top-pct FLOAT` | `5.0` | 规则 B：检测前 X% 大晶粒 |
| `--rule-b-area-frac FLOAT` | `0.30` | 规则 B：面积占比阈值 |

### SAM3 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sam3-model-id` | `facebook/sam3` | Hugging Face SAM3 模型 ID |
| `--sam3-device` | `auto` | 运行设备：`auto` / `cpu` / `cuda` / `mps` |
| `--sam3-score-threshold` | `0.5` | SAM3 实例分割分数阈值 |
| `--sam3-mask-threshold` | `0.5` | SAM3 二值 mask 阈值 |
| `--sam3-opening-disk` | `1` | 对每个 SAM3 mask 做开运算的核半径 |
| `--sam3-closing-disk` | `2` | 对每个 SAM3 mask 做闭运算的核半径 |
| `--sam3-prompt-top-ratio` | `0.05` | 从 optical labels 中按面积选前多少比例晶粒做 prompts |

---

## 算法说明

### Optical 后端

```text
彩色图像
→ 灰度化
→ 平滑去噪（gaussian / bilateral / anisotropic）
→ 中值滤波
→ CLAHE
→ Otsu 阈值
→ 形态学闭/开运算
→ 距离变换
→ peak_local_max
→ Watershed
```

### SAM3 后端

```text
输入图像
→ 读取 / 自动生成 optical labels
→ 按面积降序选取前 ceil(30%) 晶粒
→ 构建 box prompts
→ SAM3 prompt-based instance segmentation
→ 对每个返回 mask 做 opening + closing
→ 合成为最终 labels
→ 进入统一 analysis / anomaly / visualization 流程
```

### 面积法（ASTM E112 Jeffries Planimetric）

当前实现中：

```text
N_eq = N_inside + 0.5 × N_edge + 0.25 × N_corner
N_A  = N_eq / A_mm²
G    = 3.322 × log10(N_A) − 2.954
```

其中：
- `inside`：不接触边界
- `edge`：接触边界但不属于角部晶粒
- `corner`：同时接触两条相邻边

### 截线法（ASTM E112 Heyn Intercept）

测试图案为：**4 条线 + 3 个同心圆**

```text
l̄ = L_total / P
G = −6.6457 × log10(l̄_mm) − 3.298
```

交点计数会过滤长度 `< min_intercept_px` 的细小截段，以减少划痕和厚晶界带来的重复计数。

---

## 输出结构

```text
data/{image_name}/
├── optical/
│   ├── {name}_original.png
│   ├── {name}_segmented.png
│   ├── {name}_area_method.png
│   ├── {name}_intercept_method.png
│   ├── {name}_anomaly.png
│   ├── {name}_distribution.png
│   ├── {name}_labels.npy
│   └── {name}_results.json
└── sam3/
    ├── {name}_original.png
    ├── {name}_segmented.png
    ├── {name}_area_method.png
    ├── {name}_intercept_method.png
    ├── {name}_anomaly.png
    ├── {name}_distribution.png
    ├── {name}_labels.npy
    ├── {name}_results.json
    ├── {name}_sam3_prompts.json
    ├── {name}_sam3_prompts_masks.npz
    ├── {name}_sam3_prompts_raw.json
    └── {name}_sam3_prompts_raw_masks.npy
```

### `results.json` 关键字段

```json
{
  "segmentation": {
    "backend": "sam3",
    "method": "sam3_prompt_boxes",
    "details": {
      "prompt_source_labels_path": ".../optical/RG1_1_1_labels.npy",
      "prompt_top_ratio": 0.05,
      "prompt_selected_grain_ids": [1, 4, 7],
      "prompt_selected_grain_count": 3,
      "prompt_mode": "boxes_from_optical_top_area",
      "postprocess_mode": "opening_then_closing_per_mask",
      "mask_conversion": {
        "original_masks": 24,
        "morphology_processed_masks": 24,
        "kept_masks": 18,
        "labeled_grains": 18,
        "postprocess": "opening_then_closing",
        "opening_disk_size": 1,
        "closing_disk_size": 2
      },
      "sam3_device": "cpu"
    }
  }
}
```

---

## 代码结构

```text
grain-analysis/
├── README.md
├── environment.yml
├── main.py
├── scripts/
│   ├── sam3_interactive_gui.py
│   ├── labels_to_sam3_prompts.py
│   └── analyze_sam3_masks.py
└── src/
    ├── preprocessing.py
    ├── segmentation.py
    ├── analysis.py
    ├── anomaly.py
    ├── visualization.py
    ├── io_utils.py
    ├── pipeline.py
    └── sam3_backend.py
```

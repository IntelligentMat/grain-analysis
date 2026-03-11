# Grain Size Analysis System

基于传统光学图像处理与 `sam3` prompt-based 后端的金相图晶粒自动分割与尺寸分析系统，支持 ASTM E112-13 面积法、截线法和异常晶粒检测。当前 README 以仓库**已实现行为**为准，重点覆盖 CLI、YAML 配置、输出工件与重绘流程。

## 功能

- `optical` 后端：灰度化、平滑、中值滤波、CLAHE、Otsu、形态学、距离变换、Watershed
- `sam3` 后端：先复用或生成 `optical` labels，再自动挑选 prompt boxes 调用 SAM3 做实例分割
- 面积法（Planimetric / Jeffries）ASTM G 值计算
- 截线法（Heyn Intercept）ASTM G 值计算
- 异常晶粒识别：尺寸比法、长尾分布法、3σ 准则
- 可视化输出：分割图、面积法标注图、截线法标注图、异常图、尺寸分布图
- 结果工件保存：`results.json`、`*_labels.npy`、`*_grain_props.npy`
- 基于已有 `results.json` 的重绘流程：`--render-from-results`
- 严格 YAML 配置校验，键名与 `main.py` CLI 长选项保持一致

## 环境配置

```bash
micromamba env create -f environment.yml
micromamba activate grain-analysis
```

主要依赖包括：`numpy`、`scipy`、`scikit-image`、`opencv-python`、`matplotlib`、`Pillow`、`PyYAML`、`click`、`tqdm`。

如果要运行 `sam3`，还需要：

- 可用的 `torch` 运行设备：`cpu` / `cuda` / `mps`
- 可访问的 SAM3 模型权重
- 与模型来源匹配的认证环境（例如 Hugging Face 登录）

## 配置文件

仓库当前提供的 YAML 示例位于：

- `config/config.example.yml`

配置解析规则以 `src/config.py` 为准：

- 允许的配置组只有：`run`、`preprocessing`、`segmentation`、`analysis`、`sam3`
- YAML 中的键名必须与 CLI 长选项一致，例如 `pixels-per-micron`、`min-intercept-px`
- `input` 与 `render-from-results` 是 **CLI-only** 参数，不能写入 YAML
- 未知分组、未知键、错误类型不会被静默忽略，而是直接报错
- 优先级为：**内建默认值 < YAML 配置 < CLI 显式传参**

示例：

```bash
python main.py \
  --input ./steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --config ./config/config.example.yml \
  --pixels-per-micron 2.25
```

`config/config.example.yml` 当前覆盖的参数范围包括：

- `run`：`output`、`segmentation-backend`
- `preprocessing`：`smooth-mode`、`gaussian-sigma`、`median-kernel`、`clahe-clip`
- `segmentation`：`min-distance`、`closing-disk`、`opening-disk`、`min-grain-area`、`remove-border`
- `analysis`：`pixels-per-micron`、`min-intercept-px`、`rule-a-threshold`、`rule-b-top-pct`、`rule-b-area-frac`
- `sam3`：`sam3-model-id`、`sam3-device`、`sam3-score-threshold`、`sam3-mask-threshold`、`sam3-opening-disk`、`sam3-closing-disk`、`sam3-prompt-top-ratio`

## 快速运行

### 1) 单张图像：`optical` 后端

```bash
python main.py \
  --input ./steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --segmentation-backend optical \
  --pixels-per-micron 2.25
```

默认输出目录：`./data/RG36_2_1/optical/`

### 2) 单张图像：`sam3` 后端

```bash
python main.py \
  --input ./steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --segmentation-backend sam3 \
  --pixels-per-micron 2.25 \
  --sam3-device cpu
```

默认输出目录：`./data/RG36_2_1/sam3/`

说明：

- `sam3` 不是独立全自动分割入口，而是 **prompt-based** 后端
- 流程会优先查找 `data/{image_name}/optical/{image_name}_labels.npy`
- 如果 optical labels 不存在，会先自动执行一次 `optical` 分割
- 然后从 optical labels 中按面积选择前 `sam3-prompt-top-ratio` 比例的晶粒生成 box prompts，再执行 SAM3

### 3) 整个文件夹批处理

```bash
python main.py \
  --input ./steel_grain_size_dataset/RG \
  --output ./data \
  --segmentation-backend optical \
  --pixels-per-micron 2.25
```

如果改为 `--segmentation-backend sam3`，每张图会输出到各自的 `sam3/` 子目录。

### 4) 结合 YAML 配置运行

```bash
python main.py \
  --input ./steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --config ./config/config.example.yml
```

如果同时传入 CLI 参数和 YAML 中同名配置，CLI 显式参数优先。

### 5) 从已有结果重绘可视化

```bash
python main.py \
  --render-from-results ./data/RG36_2_1/optical/RG36_2_1_results.json
```

说明：

- 该模式会读取已有 `results.json` 与相关工件，重新生成可视化
- 该模式**不会重新分割**图像
- `--config` 不能与 `--render-from-results` 同时使用
- 若不传 `--output`，重绘结果默认写回原结果目录

### 6) 使用不同平滑模式

```bash
python main.py \
  --input ./steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --segmentation-backend optical \
  --pixels-per-micron 2.25 \
  --smooth-mode bilateral
```

可选平滑模式：`gaussian`、`bilateral`、`anisotropic`。

## CLI 参数说明

```bash
python main.py [OPTIONS]
```

### 输入 / 输出

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input PATH` | — | 输入图像文件或文件夹路径 |
| `--render-from-results PATH` | — | 根据已有 `results.json` 重绘可视化 |
| `--config PATH` | — | 加载 YAML 配置；仅分析模式可用 |
| `--output DIR` | `./data` | 输出根目录；重绘模式下不传则写回原目录 |

### 预处理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--smooth-mode` | `gaussian` | 平滑模式：`gaussian` / `bilateral` / `anisotropic` |
| `--gaussian-sigma FLOAT` | 自动 | 高斯滤波标准差 |
| `--median-kernel INT` | `3` | 中值滤波核大小，建议为奇数 |
| `--clahe-clip FLOAT` | `2.0` | CLAHE 对比度限制 |

### 分割参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--segmentation-backend` | `optical` | 分割后端：`optical` / `sam3` |
| `--min-distance INT` | 自动 | Watershed marker 最小间距（主要用于 `optical`） |
| `--closing-disk INT` | `2` | `optical` 形态学闭运算核半径 |
| `--opening-disk INT` | `1` | `optical` 形态学开运算核半径 |
| `--min-grain-area INT` | 自动 | 最小晶粒面积（像素²） |
| `--remove-border / --keep-border` | `--keep-border` | 是否移除接触图像边界的晶粒 |

### 物理量与分析参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pixels-per-micron FLOAT` | `1.0` | 像素/微米换算系数 |
| `--min-intercept-px INT` | `3` | 截线法最小有效截段长度 |
| `--rule-a-threshold FLOAT` | `3.0` | 规则 A：`d_max / d_avg` 阈值 |
| `--rule-b-top-pct FLOAT` | `5.0` | 规则 B：统计前 X% 大晶粒 |
| `--rule-b-area-frac FLOAT` | `0.30` | 规则 B：面积占比阈值 |

### SAM3 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sam3-model-id` | `facebook/sam3` | SAM3 模型 ID |
| `--sam3-device` | `auto` | 设备：`auto` / `cpu` / `cuda` / `mps` |
| `--sam3-score-threshold` | `0.5` | 实例分割得分阈值 |
| `--sam3-mask-threshold` | `0.5` | 二值 mask 阈值 |
| `--sam3-opening-disk` | `1` | 每个 SAM3 mask 的开运算核半径 |
| `--sam3-closing-disk` | `2` | 每个 SAM3 mask 的闭运算核半径 |
| `--sam3-prompt-top-ratio` | `0.05` | 从 optical labels 中按面积选前多少比例晶粒做 prompts |

## 算法说明

### `optical` 后端

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
→ labels
→ analysis / anomaly / visualization
```

### `sam3` 后端

```text
输入图像
→ 读取或自动生成 optical labels
→ 按面积排序并选择 top-ratio 晶粒
→ 构建 prompt boxes
→ SAM3 prompt-based instance segmentation
→ 对返回 masks 做 opening + closing
→ 合成为 labels
→ analysis / anomaly / visualization
```

### 面积法（ASTM E112 Jeffries Planimetric）

当前实现按内点、边界、角点三类晶粒计数：

```text
N_eq = N_inside + 0.5 × N_edge + 0.25 × N_corner
N_A  = N_eq / A_mm²
G    = 3.322 × log10(N_A) − 2.954
```

`results.json` 中会同时保存：

- `n_inside`、`n_edge`、`n_corner`
- `inside_grain_ids`、`edge_grain_ids`、`corner_grain_ids`

### 截线法（ASTM E112 Heyn Intercept）

当前实现使用 **4 条线 + 3 个同心圆** 的测试图样。

```text
l̄ = L_total / P
G = −6.6457 × log10(l̄_mm) − 3.298
```

其中：

- 细小截段会按 `min_intercept_px` 过滤
- `results.json` 中的 `counting_basis` 当前固定为 `grain_segments_n`
- 输出会保留 `intersected_grain_ids` 与交点坐标信息，便于重绘

### 异常晶粒判定

当前实现输出三类规则的判定结果及异常晶粒 ID：

- 规则 A：`d_max / d_avg > threshold`
- 规则 B：前 `X%` 大晶粒面积占比超过阈值
- 规则 C：`d > μ + 3σ`

## 输出结构

默认目录结构如下：

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
│   ├── {name}_grain_props.npy
│   └── {name}_results.json
└── sam3/
    ├── {name}_original.png
    ├── {name}_segmented.png
    ├── {name}_area_method.png
    ├── {name}_intercept_method.png
    ├── {name}_anomaly.png
    ├── {name}_distribution.png
    ├── {name}_labels.npy
    ├── {name}_grain_props.npy
    └── {name}_results.json
```

同一张图像的不同后端结果会分开写入 `optical/` 与 `sam3/` 子目录，不会混写。

对于 `sam3` 运行，`results.json` 的 `artifacts` 节点还会额外记录 prompt 与 raw mask 相关工件路径（若该次运行生成了这些文件）。

### `results.json` 关键字段

当前 `results.json` 的关键结构包括：

```json
{
  "image_name": "grid_fixture",
  "pixels_per_micron": 1.0,
  "measurement_mode": "physical_um",
  "config": {
    "source_path": null,
    "effective": {
      "analysis": {
        "pixels-per-micron": 1.0
      }
    },
    "cli_overrides": {
      "analysis": {
        "pixels-per-micron": 1.0
      }
    }
  },
  "artifacts": {
    "labels_path": "..._labels.npy",
    "grain_props_path": "..._grain_props.npy"
  },
  "segmentation": {
    "backend": "optical",
    "method": "watershed",
    "params": {},
    "details": {}
  },
  "grain_statistics": {
    "mean_diameter_um": 17.02,
    "mean_area_um2": 227.1,
    "mean_aspect_ratio": 1.18,
    "mean_circularity": 0.83
  },
  "area_method": {
    "n_inside": 1,
    "n_edge": 4,
    "n_corner": 4,
    "inside_grain_ids": [5]
  },
  "intercept_method": {
    "counting_basis": "grain_segments_n",
    "total_intersections": 21.0,
    "intersected_grain_ids": [1, 2, 3]
  }
}
```

重点字段说明：

- `segmentation.backend`：当前分割后端，`optical` 或 `sam3`
- `segmentation.method`：当前实现中 `optical` 为 `watershed`
- `config.effective`：合并默认值、YAML 与 CLI 后的最终配置
- `config.cli_overrides`：仅记录 CLI 显式覆盖项
- `artifacts`：分析输出与附加工件路径
- `grain_statistics`：包含粒径、面积，以及 `mean_aspect_ratio`、`mean_circularity`
- `area_method` / `intercept_method`：除统计量外，还会保留 grain ID 列表与重绘所需细节

## CI 与开发验证

仓库当前包含：

- `requirements-ci.txt`
- `requirements-dev.txt`
- `.pre-commit-config.yaml`
- `.github/workflows/ci.yml`

常用开发命令：

```bash
python -m pip install -r requirements-dev.txt
pre-commit run --all-files
python -m unittest discover -s tests -v
```

如果只想快速验证 README 中提到的核心分析流程，可优先运行：

```bash
python -m unittest tests.test_pipeline -v
python -m unittest tests.test_pipeline_helpers -v
```

这两组测试分别覆盖：

- `tests/test_pipeline.py`：`optical` 主流程、工件写出、`results.json` 关键字段
- `tests/test_pipeline_helpers.py`：后端归一化、`sam3` 路径对已有 optical labels 的复用逻辑

## 代码结构

```text
grain-analysis/
├── README.md
├── environment.yml
├── config/
│   └── config.example.yml
├── main.py
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── analysis.py
│   ├── anomaly.py
│   ├── visualization.py
│   ├── io_utils.py
│   ├── config.py
│   ├── pipeline.py
│   └── sam3_backend.py
└── tests/
    ├── test_pipeline.py
    └── test_pipeline_helpers.py
```

更详细的架构说明可参考 `docs/architecture.md`。

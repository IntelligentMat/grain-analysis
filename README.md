# Grain Size Analysis System

基于传统图像处理的金相图晶粒自动分割与尺寸分析系统，符合 ASTM E112-13 标准。

## 功能

- 自动晶粒分割（Otsu 阈值 + Watershed 分水岭）
- 面积法（Planimetric/Jeffries）ASTM G 值计算（N_A 单位 mm⁻²，结果符合 ASTM E112）
- 截线法（Heyn Intercept）ASTM E112 标准测试图案（4 条线 + 3 个同心圆）
- 异常晶粒识别（尺寸比法 / 长尾分布法 / 3σ 准则）
- 可视化标注图输出（分割图、分析图、异常图、分布直方图）

---

## 环境配置

```bash
micromamba env create -f environment.yml
micromamba activate grain-analysis
```

依赖包含 `medpy`（各向异性扩散）、`scikit-image`、`opencv`、`matplotlib`。

---

## 快速运行

### 单张图像

```bash
python main.py \
  --input /Users/siyuliu/Desktop/OM/steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --pixels-per-micron 2.25
```

输出写入 `./data/RG36_2_1/`。

### 整个文件夹（批量）

```bash
python main.py \
  --input /Users/siyuliu/Desktop/OM/steel_grain_size_dataset/RG/ \
  --pixels-per-micron 2.25 \
  --output ./data
```

### 使用双边滤波抑制划痕

```bash
python main.py \
  --input /path/to/image.jpg \
  --pixels-per-micron 2.25 \
  --smooth-mode bilateral
```

---

## CLI 参数说明

```
python main.py [OPTIONS]
```

### 必填参数

| 参数 | 缩写 | 说明 |
|---|---|---|
| `--input PATH` | `-i` | 输入图像文件或文件夹路径 |

### 通用参数

| 参数 | 缩写 | 默认值 | 说明 |
|---|---|---|---|
| `--output DIR` | `-o` | `./data` | 输出根目录 |

### 预处理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--smooth-mode` | `gaussian` | 平滑模式：`gaussian` / `bilateral`（双边滤波）/ `anisotropic`（各向异性扩散） |
| `--gaussian-sigma FLOAT` | 自动估计 | 高斯滤波标准差（仅 `smooth-mode=gaussian` 时有效） |
| `--median-kernel INT` | `3` | 中值滤波核大小（奇数），去椒盐噪声 |
| `--clahe-clip FLOAT` | `2.0` | CLAHE 对比度限制，增强晶界可见性 |

> `bilateral`：平滑晶粒内部划痕，同时保留晶界边缘；`anisotropic`：Perona-Malik 扩散，效果更强但速度较慢。

### 分割参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--min-distance INT` | 自动（图像短边 × 5%）| Watershed marker 最小间距（像素） |
| `--closing-disk INT` | `2` | 形态学闭运算核半径，填补断裂晶界 |
| `--opening-disk INT` | `1` | 形态学开运算核半径，消除细线/划痕（增大至 2~4 可增强去划痕效果） |
| `--min-grain-area INT` | 自动 | 最小晶粒面积（像素²） |
| `--remove-border` | 否 | 移除接触图像边缘的晶粒 |

### 物理单位参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pixels-per-micron FLOAT` | `1.0` | 像素/微米换算系数（500X 不锈钢 316L：2.25 px/μm） |

### 截线法参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--min-intercept-px INT` | `3` | 最小有效截段长度（px），过滤仅擦过晶粒角点的虚假截交 |

### 异常检测参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rule-a-threshold FLOAT` | `3.0` | 规则 A：d_max/d_avg 超过此值判定为异常 |
| `--rule-b-top-pct FLOAT` | `5.0` | 规则 B：检测前 X% 大晶粒的面积占比 |
| `--rule-b-area-frac FLOAT` | `0.30` | 规则 B：前 X% 晶粒面积占比超过此值判定为异常 |

---

## 算法说明

### 预处理

```
彩色图像 → 灰度化 → 平滑去噪（gaussian / bilateral / anisotropic）→ 中值滤波 → CLAHE
```

### 晶粒分割

```
预处理图 → Otsu 阈值 → 形态学闭/开运算 → 距离变换 → peak_local_max → Watershed
```

`min_distance` 控制 marker 最小间距，默认为图像短边的 5%（400×300 图约 15 px）。

### 面积法（ASTM E112 Jeffries Planimetric）

```
N_eq = N_inside + 0.5 × N_intersect
N_A  = N_eq / A_mm²            （单位：grains/mm²）
G    = 3.322 × log₁₀(N_A) − 2.954
```

### 截线法（ASTM E112 Heyn Intercept — 标准图案）

测试图案：**4 条线 + 3 个同心圆**（符合 ASTM E112 附录 A 标准）

- 水平线：图像底部（距边缘 5% margin）
- 垂直线：图像左侧（距边缘 5% margin）
- 对角线 ↘ / ↗：margin 内缩
- 同心圆：圆心居中，半径比例 0.7958 / 0.5305 / 0.2653 × min(H,W)/2

```
l̄ = L_total / P          （P = 总交点数，L_total = 测试路径总长）
G = −6.6457 × log₁₀(l̄_mm) − 3.298
```

交点计数过滤掉长度 < `min_intercept_px` 的细小截段，避免划痕和厚晶界导致的重复计数。

---

## 输出结构

```
data/{image_name}/
├── {name}_original.png          # 原始输入图像
├── {name}_segmented.png         # 分割结果（伪彩色晶粒叠加）
├── {name}_area_method.png       # 面积法标注图
├── {name}_intercept_method.png  # 截线法标注图（ASTM 4线+3圆，红点为交点）
├── {name}_anomaly.png           # 异常晶粒高亮图（红色标注）
├── {name}_distribution.png      # 晶粒尺寸分布直方图
└── {name}_results.json          # 完整分析结果（JSON）
```

### results.json 关键字段

```json
{
  "grain_statistics": {
    "count": 35,
    "mean_diameter_px": 45.2,
    "std_diameter_px": 12.1
  },
  "area_method": {
    "n_inside": 27,
    "n_intersect": 13,
    "n_equivalent": 33.5,
    "n_a_per_mm2": 1059.96,
    "astm_g_value": 7.10,
    "mean_grain_area_mm2": 9.44e-4,
    "mean_diameter_um": 34.66
  },
  "intercept_method": {
    "n_lines": 4,
    "n_circles": 3,
    "total_intersections": 63,
    "total_line_length_px": 3053,
    "mean_intercept_length_um": 48.5,
    "astm_g_value": 5.14
  },
  "anomaly_detection": {
    "has_anomaly": true,
    "total_anomalous_grains": 2
  }
}
```

---

## 代码结构

```
grain-analysis/
├── main.py              # CLI 入口（click）
├── environment.yml      # micromamba 环境配置
└── src/
    ├── preprocessing.py # 灰度化、平滑去噪（高斯/双边/各向异性）、CLAHE
    ├── segmentation.py  # Otsu 阈值 + Watershed 分水岭分割
    ├── analysis.py      # regionprops 特征提取、面积法、截线法（ASTM 图案）
    ├── anomaly.py       # 三规则异常晶粒判定
    ├── visualization.py # 标注图生成（matplotlib）
    ├── io_utils.py      # 图像读取、目录管理、JSON 存储
    └── pipeline.py      # 端到端流程编排
```

---

## 测试数据集

`steel_grain_size_dataset/RG/`：480 张不锈钢 316L 光学显微镜图像
- 格式：JPG，400×300 px
- 分辨率：2.25 px/μm（500X）
- 参考分割标注：`steel_grain_size_dataset/RGMask/`

---

## 参考标准与资料

- ASTM E112-13: 金属平均晶粒度测定方法
- GBT/6394-2017: 金属平均晶粒度测定方法
- [grain-size-analysis-metallic-materials](https://github.com/rricc22/grain-size-analysis-metallic-materials)：分水岭 + ASTM 参考实现
- [grain-size-analysis-tools (NIST GSAT)](https://github.com/usnistgov/grain-size-analysis-tools)：截线法标准图案参考实现

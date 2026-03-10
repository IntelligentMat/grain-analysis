# 晶粒自动化分析系统 — 算法架构图

## 系统总览

```mermaid
flowchart TD
    INPUT["📂 输入\n图像路径 / 文件夹"]

    subgraph PRE["模块 1：预处理 preprocessing.py"]
        P1["灰度化\nRGB → Gray"]
        P2["降噪\n高斯滤波 σ=3\n中值滤波 3×3"]
        P3["对比度增强\nCLAHE\nclip=2.0, tile=(8,8)"]
        P1 --> P2 --> P3
    end

    subgraph SEG["模块 2：晶粒分割 segmentation.py"]
        S1["Otsu 自适应阈值\n→ 二值图（晶界为亮）"]
        S2["形态学操作\n闭运算（fill holes）\n开运算（remove noise）"]
        S3["距离变换\ndistance_transform_edt"]
        S4["局部极大值检测\nmin_distance=50\n→ Watershed Markers"]
        S5["Watershed 分割"]
        S6["后处理\n移除小区域 < min_grain_area\n移除边界接触区域（可选）"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end

    subgraph FEAT["模块 3：特征提取 analysis.py"]
        F1["skimage.measure.regionprops"]
        F2["面积 area\n周长 perimeter\n等效直径 equivalent_diameter"]
        F3["长宽比 aspect_ratio\n圆度 circularity\n质心 centroid\n边界框 bbox"]
        F4["像素 → μm 换算\n× 1/pixels_per_micron"]
        F1 --> F2 & F3 --> F4
    end

    subgraph AREA["模块 4a：面积法 Planimetric Method"]
        A1["定义测量区域\n默认全图矩形"]
        A2["晶粒计数\nN_inside（权重 1）\nN_edge（权重 1/2）\nN_corner（权重 1/4）"]
        A3["等效晶粒数\nN_eq = N_inside + 0.5·N_edge + 0.25·N_corner"]
        A4["晶粒密度\nN_A = N_eq / 面积(mm²)"]
        A5["ASTM G 值\nG = 3.322·log₁₀(N_A) - 2.954"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph INTC["模块 4b：截线法 Heyn Intercept Method"]
        I1["绘制 ASTM E112 测试图样\n4 条测试线（水平/垂直/对角）\n3 个同心圆"]
        I2["路径遍历 & 计段\n开放路径：端段×0.5，内段×1.0\n闭合路径：所有段×1.0"]
        I3["截距密度\nN_L = N_total / L_total (px⁻¹)"]
        I4["平均截距长度\nl̄ = 1/N_L → px → μm → mm"]
        I5["ASTM G 值\nG = -6.6457·log₁₀(l̄) - 3.298"]
        I1 --> I2 --> I3 --> I4 --> I5
    end

    subgraph ANOM["模块 5：异常晶粒判定 anomaly.py"]
        R1["规则 A：尺寸比法\nd_max / d_avg > 3.0\n→ 标记异常大晶粒"]
        R2["规则 B：长尾分布法\n前 5% 晶粒面积占比 > 30%\n→ 分布异常"]
        R3["规则 C：3σ 准则\nd > μ + 3σ\n→ 统计显著异常"]
        UNION["三规则取并集\n→ 最终异常晶粒集合"]
        R1 & R2 & R3 --> UNION
    end

    subgraph VIS["模块 6：可视化 visualization.py"]
        V1["original.png\n原始输入图像"]
        V2["segmented.png\n伪彩色晶粒标注"]
        V3["area_method.png\n测量区域 + 计数标注"]
        V4["intercept_method.png\n测试线 + 交点红点"]
        V5["anomaly.png\n异常晶粒红色高亮"]
        V6["distribution.png\n晶粒尺寸分布直方图"]
    end

    subgraph OUT["模块 7：输出 io_utils.py"]
        O1["目录结构\ndata/{image_name}/"]
        O2["results.json\n分割参数 + 统计结果\n面积法 + 截线法 G 值\n异常判定详情"]
    end

    INPUT --> PRE
    PRE --> SEG
    SEG --> FEAT
    FEAT --> AREA & INTC & ANOM
    AREA --> VIS
    INTC --> VIS
    ANOM --> VIS
    VIS --> OUT
    AREA --> OUT
    INTC --> OUT
    ANOM --> OUT
```

---

## 数据流详图

```mermaid
flowchart LR
    subgraph DATA["数据层"]
        IMG["原始图像\nJPG 400×300px\n2.25px/μm"]
        LABEL["Label 图\nnumpy uint32\n每晶粒一整数ID"]
        PROPS["晶粒属性表\nDataFrame/list[dict]"]
    end

    subgraph ALGO["算法层"]
        direction TB
        GRAY["灰度图\nuint8"]
        ENHANCED["增强灰度图\nuint8"]
        BINARY["二值图\nbool"]
        DIST["距离变换图\nfloat32"]
        MARKERS["Markers\nuint32"]

        GRAY --> ENHANCED --> BINARY --> DIST --> MARKERS --> LABEL
    end

    subgraph RESULT["结果层"]
        AREA_R["面积法结果\nN_A, G_area"]
        INTC_R["截线法结果\nN_L, l̄, G_intercept"]
        ANOM_R["异常结果\nanomaly_ids, flags"]
        JSON["results.json"]
        IMGS["标注图像 ×6"]
    end

    IMG --> GRAY
    LABEL --> PROPS
    PROPS --> AREA_R & INTC_R & ANOM_R
    AREA_R & INTC_R & ANOM_R --> JSON & IMGS
```

---

## 模块依赖关系

```mermaid
graph LR
    main["main.py\nCLI 入口\nclick"]
    pipeline["pipeline.py\n流程编排"]
    prep["preprocessing.py"]
    seg["segmentation.py"]
    ana["analysis.py"]
    anom["anomaly.py"]
    vis["visualization.py"]
    io["io_utils.py"]

    main --> pipeline
    pipeline --> prep --> seg --> ana
    ana --> anom
    ana --> vis
    anom --> vis
    vis --> io
    ana --> io

    style main fill:#4A90D9,color:#fff
    style pipeline fill:#7B68EE,color:#fff
    style prep fill:#5BA85F,color:#fff
    style seg fill:#5BA85F,color:#fff
    style ana fill:#E8A838,color:#fff
    style anom fill:#D95B5B,color:#fff
    style vis fill:#9B59B6,color:#fff
    style io fill:#95A5A6,color:#fff
```

---

## ASTM E112 测试图样（截线法）

```
┌─────────────────────────────────┐
│  ╲                           ╱  │  ← 对角线（左上→右下，右上→左下）
│    ╲        ┌─────┐        ╱    │
│      ╲    ┌─┘     └─┐    ╱      │
│        ╲  │  ┌───┐  │  ╱        │
│          ╲│  │ · │  │╱          │  ← 3 个同心圆（r = 0.7958/0.5305/0.2653）
│          ╱│  │   │  │╲          │
│        ╱  │  └───┘  │  ╲        │
│      ╱    └─┐     ┌─┘    ╲      │
│    ╱        └─────┘        ╲    │
│  ╱                           ╲  │
├─────────────────────────────────┤  ← 水平测试线（底部）
│                                 │
└─────────────────────────────────┘
▲
│  ← 垂直测试线（左侧）
```

---

## 异常判定逻辑

```mermaid
flowchart TD
    START["输入：所有晶粒直径列表"]

    RA{"规则 A\nd_max / d_avg > 3.0?"}
    RB{"规则 B\nTop5% 面积占比 > 30%?"}
    RC{"规则 C\n∃ d > μ + 3σ?"}

    AA["标记：超大晶粒 IDs"]
    AB["标记：分布异常 flag"]
    AC["标记：统计异常 IDs"]

    UNION["并集合并\nhas_anomaly = A OR B OR C"]
    OUTPUT["输出：anomaly_detection 字段"]

    START --> RA & RB & RC
    RA -->|Yes| AA
    RB -->|Yes| AB
    RC -->|Yes| AC
    AA & AB & AC --> UNION --> OUTPUT
```

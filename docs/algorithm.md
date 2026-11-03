# 钱币定位系统：自定义算法说明与最终实验结果

## 1. 实验目标与测试集

系统输入一张包含一个或多个钱币的图片，输出每个钱币的圆心坐标
`(x, y)` 和半径 `r`，并在图片上绘制拟合圆。

测试集分为两组，避免高噪声背景影响普通场景的主指标：

| 目录 | 内容 | 图片数 |
|---|---|---:|
| `val/normal/` | 普通背景 | 11 |
| `val/noisy/` | 高频纹理背景及困难样本（101、111、121、131） | 4 |

`val/2.jpg` 和 `val/3.jpg` 因像素尺寸与其他样本不同而删除。当前系统不把
硬币重叠、光影过暗和严重倾斜作为主要评价条件。

## 2. 整体流程

```text
输入图像
    ↓
缩放到最长边 800 像素
    ↓
BGR → 灰度图
    ↓
高斯滤波（普通组 11；高噪组 17）
    ↓
自定义 Canny
  1. Sobel 梯度
  2. 梯度幅值和方向
  3. 四方向量化
  4. 非极大值抑制
  5. 双阈值分类
  6. 滞后连接
    ↓
自定义 Hough Circle
  1. 读取边缘点和梯度方向
  2. 沿梯度正、反方向估计圆心
  3. 对半径范围进行投票
  4. 保留超过投票阈值的候选
  5. 按投票数排序
  6. 按圆心最小距离进行非极大值抑制
    ↓
坐标缩放回原图
    ↓
输出圆心、半径、拟合图、边缘图和 JSON 报告
```

自定义实现不调用 `cv2.Canny` 或 `cv2.HoughCircles`。OpenCV 只用于图像读写、
缩放、灰度转换、高斯滤波和结果绘制。

## 3. 函数说明

### 3.1 `custom_canny.py`

#### `_non_maximum_suppression(magnitude, angle)`

沿每个像素的梯度方向比较前后邻居，只保留局部最大梯度，使粗边缘变为较窄的
候选边缘。

| 参数 | 类型 | 说明 |
|---|---|---|
| `magnitude` | `np.ndarray` | 梯度幅值图 |
| `angle` | `np.ndarray` | 梯度方向图，单位为弧度 |

返回非极大值抑制后的梯度幅值图。

#### `canny(gray, *, low_threshold, high_threshold)`

完成自定义 Canny 的完整流程。首先用 Sobel 核计算水平和垂直梯度，再进行方向
量化与非极大值抑制；高于高阈值的像素作为强边缘，介于高低阈值之间的像素作为
弱边缘，最后只连接到强边缘的弱边缘。

| 参数 | 类型 | 说明 |
|---|---|---|
| `gray` | `np.ndarray` | 二维灰度图 |
| `low_threshold` | `float` | 弱边缘下限，低于它的像素丢弃 |
| `high_threshold` | `float` | 强边缘阈值，必须不小于低阈值 |

返回三个数组：

- `edges`：0/255 二值边缘图；
- `gradient_x`：水平 Sobel 梯度；
- `gradient_y`：垂直 Sobel 梯度。

当前流水线使用 `low_threshold = param1 / 2`、`high_threshold = param1`。
这只是实验映射，不代表与 OpenCV Canny 的内部阈值完全等价。

### 3.2 `custom_hough.py`

#### `DetectedCircle`

保存一个候选圆：

| 字段 | 类型 | 说明 |
|---|---|---|
| `x` | `float` | 当前缩放图像中的圆心横坐标 |
| `y` | `float` | 当前缩放图像中的圆心纵坐标 |
| `radius` | `float` | 半径 |
| `votes` | `int` | 累加器投票数 |

#### `hough_circles(edges, gradient_x, gradient_y, *, ...)`

使用边缘点的梯度方向估计圆心。对于每个边缘点和每个候选半径，分别沿梯度正、
反方向投票。完成全部半径搜索后，候选按票数从高到低排序，并以 `min_dist`
抑制相近圆心。

| 参数 | 类型 | 说明 |
|---|---|---|
| `edges` | `np.ndarray` | 0/255 二值边缘图 |
| `gradient_x` | `np.ndarray` | 水平梯度，必须与边缘图同尺寸 |
| `gradient_y` | `np.ndarray` | 垂直梯度，必须与边缘图同尺寸 |
| `min_radius` | `int` | 最小搜索半径 |
| `max_radius` | `int` | 最大搜索半径 |
| `radius_step` | `int` | 半径步长，越小越精确但越慢 |
| `vote_threshold` | `int` | 自定义累加器的最低票数 |
| `min_dist` | `float` | 已选圆心之间的最小距离 |
| `dp` | `float` | 圆心累加器缩放比例；大于 1 可合并邻近投票 |

`vote_threshold` 不是 OpenCV 的 `param2`。当前流水线用近似关系
`max(2, round(param2 / 4))` 转换，实际效果需单独校准。

### 3.3 `custom_pipeline.py`

#### `detect_circles(image, **parameters)`

负责图像预处理、调用自定义 Canny 和 Hough Circle，并把检测坐标换算回原图。

主要参数：

| 参数 | 说明 |
|---|---|
| `max_dimension` | 缩放图像最长边上限 |
| `blur_kernel` | 高斯核，必须是大于等于 3 的奇数 |
| `param1` | 自定义 Canny 的强边缘阈值 |
| `param2` | 参考投票阈值，内部映射为自定义票数 |
| `min_radius` / `max_radius` | 原图坐标下的半径范围 |
| `min_dist` | 原图坐标下的圆心最小距离 |
| `dp` | Hough 圆心累加器缩放比例 |
| `multi_scale` | 是否额外融合另一高斯尺度的边缘 |

返回原图坐标下的圆列表、实际检测图和边缘图。

#### `process_file(path, output_dir, parameters)`

读取一张图片，调用检测流程，保存：

- 拟合标注图：`<name>.jpg`
- 边缘图：`<name>_edges.png`
- 结果由批处理入口汇总到 `report.json`

#### `main(argv=None)`

解析命令行参数，识别输入文件或目录中的 JPG 图片，并批量调用
`process_file`。

运行示例：

```bash
uv run cv1-custom val/normal -o results/final/custom-normal \
  --blur-kernel 11 --param2 40
```

## 4. 参数对最终定位结果的影响

| 参数 | 增大后的主要效果 | 可能副作用 |
|---|---|---|
| `blur_kernel` | 抑制高频纹理和孤立边缘 | 钱币外轮廓变弱，小钱币可能漏检 |
| `param1` | 减少弱梯度边缘 | 圆弧断裂，投票不足 |
| `param2` | 提高候选圆的投票要求 | 减少误检，但容易漏掉不完整圆 |
| `min_radius` | 排除小尺寸伪圆 | 真实小钱币也可能被排除 |
| `max_radius` | 限制大圆搜索和大范围伪圆 | 大钱币或近景目标可能漏检 |
| `min_dist` | 减少同一钱币的重复圆 | 相邻钱币过近时可能只保留一个 |
| `radius_step` | 减小时提高半径精度 | Hough 计算时间增加 |
| `dp` | 大于 1 时合并邻近圆心投票 | 过大可能合并相邻伪圆 |
| `multi_scale` | 提高不同模糊尺度下的边缘覆盖 | 边缘和伪圆数量可能同时增加 |

当前调参结论：

- 普通组 OpenCV 对照：`blur_kernel=9、param2=50`；
- 普通组自定义算法：`blur_kernel=11、param2=40`；
- 高噪组：高斯核提高到 `17`，但不与普通组混合统计；
- `min_radius=120`、`min_dist=180`、`max_radius=400` 是两组共用的尺寸先验；
- `dp=1.0` 为默认值，`dp=1.1` 仅作为投票容差实验参数。

## 5. 最终拟合结果

以下结果由当前参数重新生成，标注图中的绿色圆为拟合圆，红点为圆心，黄色文字
给出圆心和半径。结果目录已保存在项目的 `results/final/`：

### 普通背景组

![普通背景自定义拟合结果](../results/final/custom-normal/1.jpg)

命令：

```bash
uv run cv1-custom val/normal -o results/final/custom-normal \
  --blur-kernel 11 --param2 40
```

### 高噪声组

![高噪声自定义拟合结果](../results/final/custom-noisy/101.jpg)

命令：

```bash
uv run cv1-custom val/noisy -o results/final/custom-noisy \
  --blur-kernel 17 --param2 40
```

最终批量结果：

| 组别 | 实现 | 图片数 | 总检测数 | 平均耗时 |
|---|---|---:|---:|---:|
| 普通背景 | OpenCV | 11 | 55 | 14.4 ms/张 |
| 普通背景 | 自定义 | 11 | 54 | 198.4 ms/张 |
| 高噪声 | OpenCV | 4 | 18 | 15.3 ms/张 |
| 高噪声 | 自定义 | 4 | 14 | 217.9 ms/张 |

这里的“检测数”是候选圆数量，不等同于准确率；当前数据没有逐枚人工标注的
真值坐标，因此不能据此计算严格 Precision/Recall。最终图和 `report.json` 可用于
人工核验圆心、半径和误检情况。

## 6. 结果评价与局限

算法流程、函数职责和参数含义已经拆分清楚；自定义 Canny 和 Hough Circle 也已
独立替代 OpenCV 对应函数。普通组中两种实现的候选数量接近，但自定义实现速度
较慢，原因是 Python/NumPy 按半径逐层建立累加器并进行散点投票。

OpenCV 的边缘通常更连续，因此在相同参数下不应期待自定义边缘图逐像素一致。
自定义算法当前的主要误差来源是梯度幅值定义、四方向非极大值抑制、整数化圆心
投票以及自定义票数和 OpenCV `param2` 不同。高噪声组应作为鲁棒性结果单独报告，
不能用提高平滑强度换取普通组指标。

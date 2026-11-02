# 钱币定位系统

## OpenCV 基线

当前先使用 OpenCV 建立对照基线。该阶段允许调用 `cv2.Canny` 和
`cv2.HoughCircles`，后续自定义算法会使用相同输入和输出格式进行比较。

安装依赖并运行普通背景测试集：

```bash
uv sync
uv run cv1-opencv-baseline val/normal -o results/opencv-normal
uv run cv1-custom val/normal -o results/custom-normal
```

测试图片按两类分开保存：

- `val/normal/`：普通背景；
- `val/noisy/`：背景高噪声（`101/111/121/131`）。

困难测试组不混入普通背景主指标。高噪声组使用较强平滑：

```bash
uv run cv1-opencv-baseline val/noisy -o results/opencv-noisy --blur-kernel 17
uv run cv1-custom val/noisy -o results/custom-noisy --blur-kernel 17
```

输出内容：

- `results/opencv/*.jpg`：缩放后的检测可视化图片，绿色圆轮廓表示检测结果，红点表示圆心。
- `results/opencv/*_edges.png`：与 Hough 检测使用相同缩放、高斯平滑和阈值的 Canny 二值边缘图。
- `results/opencv/report.json`：每张图片的检测数量、圆心坐标、半径和耗时。

普通背景默认参数设置为 `max-dimension=800`、`blur-kernel=11`、
`param2=32`、`min-radius=120`、`max-radius=400`。高斯平滑用于在
Hough 投票前抑制织物的短周期纹理；它属于 OpenCV 基线的预处理，不是
后续自定义 Hough 实现之外的额外检测器。可以通过命令行参数调整：

```bash
uv run cv1-opencv-baseline val --blur-kernel 15 --param2 60 --min-dist 180
```

坐标和半径会换算回原始图像尺寸，坐标原点位于左上角，`x` 向右、`y` 向下。

## 自定义算法

使用 `cv1-custom` 可以在相同预处理和参数下运行自行实现的 Canny
（Sobel、非极大值抑制、双阈值和滞后连接）与梯度引导 Hough Circle：

```bash
uv run cv1-custom val -o results/custom
```

自定义版本同样输出标注图、`*_edges.png` 和 `report.json`。它不调用
`cv2.Canny` 或 `cv2.HoughCircles`；`param2` 在内部按自定义累加器的票数尺度
进行近似映射，因此结果应与 OpenCV 基线进行实验对照，而不是期待逐像素完全一致。
默认融合 `blur-kernel=11` 和额外的 15 像素平滑尺度，以改善普通背景的
边缘保留；高噪声组建议显式使用 `--blur-kernel 17`。自定义版本也可融合额外
平滑尺度，以改善 `101.jpg` 这类
高噪声图像；使用 `--no-multi-scale` 可关闭该处理。
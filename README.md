# 钱币定位系统

## OpenCV 基线

当前先使用 OpenCV 建立对照基线。该阶段允许调用 `cv2.Canny` 和
`cv2.HoughCircles`，后续自定义算法会使用相同输入和输出格式进行比较。

安装依赖并运行 `val/` 目录下的全部测试图片：

```bash
uv sync
uv run cv1-opencv-baseline val -o results/opencv
```

输出内容：

- `results/opencv/*.jpg`：缩放后的检测可视化图片，绿色圆轮廓表示检测结果，红点表示圆心。
- `results/opencv/report.json`：每张图片的检测数量、圆心坐标、半径和耗时。

默认参数针对当前测试集设置为 `max-dimension=800`、`blur-kernel=21`、
`param2=25`、`min-radius=80`、`max-radius=400`。高斯平滑用于在
Hough 投票前抑制织物的短周期纹理；它属于 OpenCV 基线的预处理，不是
后续自定义 Hough 实现之外的额外检测器。可以通过命令行参数调整：

```bash
uv run cv1-opencv-baseline val --blur-kernel 15 --param2 60 --min-dist 180
```

坐标和半径会换算回原始图像尺寸，坐标原点位于左上角，`x` 向右、`y` 向下。
"""Gradient-guided Hough circle detection implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectedCircle:
    """自定义 Hough 投票得到的圆及其支持票数。"""

    x: float
    y: float
    radius: float
    votes: int


def hough_circles(
    edges: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    *,
    min_radius: int,
    max_radius: int,
    radius_step: int = 2,
    vote_threshold: int = 32,
    min_dist: float = 180.0,
) -> list[DetectedCircle]:
    """沿边缘梯度正反方向投票，再进行圆心非极大值抑制。

    参数:
        edges: 0/255 二值边缘图，非零像素作为投票边缘点。
        gradient_x: 与边缘图同尺寸的水平梯度图。
        gradient_y: 与边缘图同尺寸的垂直梯度图。
        min_radius: 搜索的最小圆半径，单位为当前缩放图像像素。
        max_radius: 搜索的最大圆半径，单位为当前缩放图像像素。
        radius_step: 半径搜索步长；越小越精确，但计算量越大。
        vote_threshold: 保留圆心候选所需的最少投票数。它是自定义
            累加器的票数，不等价于 OpenCV 的 `param2`。
        min_dist: 已选圆心之间的最小距离，过近的候选会被抑制。

    返回:
        按半径从大到小排列的检测圆列表，每个元素包含圆心、半径和票数。
    """
    if edges.ndim != 2 or gradient_x.shape != edges.shape or gradient_y.shape != edges.shape:
        raise ValueError("edge and gradient arrays must have matching shapes")
    if min_radius < 1 or max_radius < min_radius or radius_step < 1:
        raise ValueError("invalid radius range")

    ys, xs = np.nonzero(edges > 0)
    if not len(xs):
        return []
    gx, gy = gradient_x[ys, xs], gradient_y[ys, xs]
    length = np.hypot(gx, gy)
    valid = length > 1e-6
    xs, ys, gx, gy, length = xs[valid], ys[valid], gx[valid], gy[valid], length[valid]
    gx, gy = gx / length, gy / length
    height, width = edges.shape
    candidates: list[DetectedCircle] = []
    for radius in range(min_radius, max_radius + 1, radius_step):
        accumulator = np.zeros((height, width), dtype=np.uint16)
        for sign in (-1.0, 1.0):
            centers_x = np.rint(xs + sign * radius * gx).astype(np.int32)
            centers_y = np.rint(ys + sign * radius * gy).astype(np.int32)
            inside = (
                (centers_x >= 0)
                & (centers_x < width)
                & (centers_y >= 0)
                & (centers_y < height)
            )
            np.add.at(accumulator, (centers_y[inside], centers_x[inside]), 1)
        peak_y, peak_x = np.nonzero(accumulator >= vote_threshold)
        for y, x in zip(peak_y, peak_x):
            candidates.append(DetectedCircle(float(x), float(y), float(radius), int(accumulator[y, x])))

    candidates.sort(key=lambda circle: circle.votes, reverse=True)
    selected: list[DetectedCircle] = []
    min_dist_squared = min_dist * min_dist
    for candidate in candidates:
        if all(
            (candidate.x - chosen.x) ** 2 + (candidate.y - chosen.y) ** 2
            >= min_dist_squared
            for chosen in selected
        ):
            selected.append(candidate)
    return sorted(selected, key=lambda circle: circle.radius, reverse=True)

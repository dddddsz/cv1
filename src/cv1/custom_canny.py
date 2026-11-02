"""A small NumPy implementation of the Canny edge detector."""

from __future__ import annotations

import cv2
import numpy as np


def _non_maximum_suppression(
    magnitude: np.ndarray, angle: np.ndarray
) -> np.ndarray:
    """沿梯度方向保留局部最大梯度。

    参数:
        magnitude: 每个像素的梯度幅值。
        angle: 每个像素的梯度方向，单位为弧度。

    返回:
        非极大值抑制后的梯度幅值图。
    """
    height, width = magnitude.shape
    result = np.zeros_like(magnitude)
    direction = (np.rad2deg(angle) % 180).astype(np.uint8)
    center = magnitude[1:-1, 1:-1]
    neighbors = np.zeros((2, *center.shape), dtype=magnitude.dtype)

    masks = (
        (direction[1:-1, 1:-1] < 22) | (direction[1:-1, 1:-1] >= 157),
        (direction[1:-1, 1:-1] < 67) & (direction[1:-1, 1:-1] >= 22),
        (direction[1:-1, 1:-1] < 112) & (direction[1:-1, 1:-1] >= 67),
        (direction[1:-1, 1:-1] < 157) & (direction[1:-1, 1:-1] >= 112),
    )
    pairs = (
        (magnitude[1:-1, 2:], magnitude[1:-1, :-2]),
        (magnitude[2:, :-2], magnitude[:-2, 2:]),
        (magnitude[2:, 1:-1], magnitude[:-2, 1:-1]),
        (magnitude[2:, 2:], magnitude[:-2, :-2]),
    )
    for mask, (forward, backward) in zip(masks, pairs):
        neighbors[0][mask] = forward[mask]
        neighbors[1][mask] = backward[mask]
    keep = (center >= neighbors[0]) & (center >= neighbors[1])
    result[1:-1, 1:-1][keep] = center[keep]
    return result


def canny(
    gray: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用 Sobel、非极大值抑制和滞后连接提取边缘。

    参数:
        gray: 二维灰度图，数值可以是 uint8 或浮点类型。
        low_threshold: 弱边缘阈值；低于该值的像素会被丢弃。
        high_threshold: 强边缘阈值；达到该值的像素作为滞后连接起点。
            当前流水线中分别对应 OpenCV Canny 的 `param1 / 2` 和 `param1`。

    返回:
        edges: 0/255 二值边缘图。
        gradient_x: 水平方向 Sobel 梯度。
        gradient_y: 垂直方向 Sobel 梯度。
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a two-dimensional image")
    if not 0 <= low_threshold <= high_threshold:
        raise ValueError("thresholds must satisfy 0 <= low <= high")

    image = gray.astype(np.float32, copy=False)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = kernel_x.T
    gradient_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)
    magnitude = np.hypot(gradient_x, gradient_y)
    suppressed = _non_maximum_suppression(magnitude, np.arctan2(gradient_y, gradient_x))

    strong = suppressed >= high_threshold
    weak = suppressed >= low_threshold
    edges = strong.copy()
    pending = list(zip(*np.nonzero(strong)))
    while pending:
        y, x = pending.pop()
        y0, y1 = max(0, y - 1), min(edges.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(edges.shape[1], x + 2)
        connected = weak[y0:y1, x0:x1] & ~edges[y0:y1, x0:x1]
        for dy, dx in zip(*np.nonzero(connected)):
            yy, xx = y0 + dy, x0 + dx
            edges[yy, xx] = True
            pending.append((yy, xx))
    return (edges.astype(np.uint8) * 255, gradient_x, gradient_y)

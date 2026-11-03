"""Coin localization pipeline using the hand-written Canny and Hough stages."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from .custom_canny import canny
from .custom_hough import hough_circles
from .opencv_baseline import Circle, ImageResult, annotate


def detect_circles(image, **parameters: object) -> tuple[list[Circle], object, object]:
    """使用自定义边缘和圆检测，保持 OpenCV 基线的坐标约定。

    参数:
        image: OpenCV 读取的 BGR 彩色图像。
        parameters: 流水线参数字典，包含:
            max_dimension: 缩放图像的最长边上限。
            blur_kernel: 高斯平滑核大小，必须为不小于 3 的奇数。
            param1: 自定义 Canny 的强边缘阈值。
            min_radius/max_radius: 原图坐标下的半径搜索范围。
            param2: 参考 OpenCV 的投票阈值，内部映射为自定义票数。
            min_dist: 原图坐标下的圆心最小距离。
            dp: 自定义圆心累加器缩放比例，用于合并相邻像素投票。
            multi_scale: 是否融合当前核和当前核加 4 的两个平滑尺度边缘，
                改善高噪声图像。

    返回:
        circles: 原始图像坐标下的圆心和半径列表。
        resized: 实际参与检测的缩放图像。
        edges: 自定义 Canny 输出的二值边缘图。
    """
    height, width = image.shape[:2]
    max_dimension = int(parameters["max_dimension"])
    scale = min(1.0, max_dimension / max(height, width))
    resized = cv2.resize(image, (round(width * scale), round(height * scale))) if scale < 1 else image
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur_kernel = int(parameters["blur_kernel"])
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    param1 = float(parameters["param1"])
    edges, gx, gy = canny(blurred, low_threshold=param1 / 2, high_threshold=param1)
    if bool(parameters["multi_scale"]):
        secondary = cv2.GaussianBlur(gray, (blur_kernel + 4, blur_kernel + 4), 0)
        secondary_edges, secondary_gx, secondary_gy = canny(
            secondary,
            low_threshold=param1 / 2,
            high_threshold=param1,
        )
        edges = edges | secondary_edges
        gx = (gx + secondary_gx) / 2
        gy = (gy + secondary_gy) / 2
    circles = hough_circles(
        edges,
        gx,
        gy,
        min_radius=round(int(parameters["min_radius"]) * scale),
        max_radius=round(int(parameters["max_radius"]) * scale),
        # 自定义累加器的票数尺度与 OpenCV param2 不同，使用同名参数做近似映射。
        vote_threshold=max(2, round(int(parameters["param2"]) / 4)),
        min_dist=float(parameters["min_dist"]) * scale,
        dp=float(parameters["dp"]),
    )
    return [
        Circle(c.x / scale, c.y / scale, c.radius / scale) for c in circles
    ], resized, edges


def process_file(path: Path, output_dir: Path, parameters: dict[str, object]) -> ImageResult:
    """处理单图并保存自定义边缘图、标注图和检测结果。

    参数:
        path: 输入 JPG 图片路径。
        output_dir: 输出标注图、边缘图和报告的目录。
        parameters: 传递给自定义检测流水线的参数字典。

    返回:
        包含文件名、处理耗时和检测圆列表的结果对象。
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {path}")
    started = time.perf_counter()
    circles, resized, edges = detect_circles(image, **parameters)
    elapsed_ms = (time.perf_counter() - started) * 1000
    scale = resized.shape[1] / image.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / path.name), annotate(resized, circles, scale))
    cv2.imwrite(str(output_dir / f"{path.stem}_edges.png"), edges)
    return ImageResult(path.name, elapsed_ms, circles)


def main(argv: list[str] | None = None) -> None:
    """批量运行自定义 Canny/Hough 管线。

    命令行输入可以是单张 JPG 或包含 JPG 图片的目录；结果会同时保存
    为可视化图片、Canny 边缘图和 JSON 报告。
    """
    parser = argparse.ArgumentParser(description="Custom coin localization pipeline")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("results/custom"))
    parser.add_argument("--min-dist", type=float, default=180.0)
    parser.add_argument("--dp", type=float, default=1.0)
    parser.add_argument("--param1", type=float, default=100.0)
    parser.add_argument("--param2", type=int, default=32)
    parser.add_argument("--min-radius", type=int, default=120)
    parser.add_argument("--max-radius", type=int, default=400)
    parser.add_argument("--max-dimension", type=int, default=800)
    parser.add_argument("--blur-kernel", type=int, default=11)
    parser.add_argument(
        "--no-multi-scale",
        action="store_false",
        dest="multi_scale",
        help="只使用单一高斯平滑尺度",
    )
    parser.set_defaults(multi_scale=True)
    args = parser.parse_args(argv)
    paths = sorted(args.input.glob("*.jpg")) if args.input.is_dir() else [args.input]
    parameters = vars(args).copy()
    parameters.pop("input")
    parameters.pop("output")
    results = [process_file(path, args.output, parameters) for path in paths]
    (args.output / "report.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    for result in results:
        print(f"{result.image}: {len(result.circles)} circle(s), {result.elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()

"""OpenCV baseline for coin localization.

This module is intentionally kept separate from the later hand-written
edge and circle detectors so that the two implementations can be compared.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class ImageResult:
    image: str
    elapsed_ms: float
    circles: list[Circle]


def detect_circles(
    image: np.ndarray,
    *,
    dp: float = 1.2,
    min_dist: float = 180.0,
    param1: float = 100.0,
    param2: float = 32.0,
    min_radius: int = 120,
    max_radius: int = 400,
    max_dimension: int = 800,
    blur_kernel: int = 11,
) -> tuple[list[Circle], np.ndarray]:
    """预处理图像并调用 OpenCV HoughCircles，返回原图坐标下的圆。"""
    if image is None or image.ndim not in (2, 3):
        raise ValueError("image must be a grayscale or color image")

    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        resized = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    if blur_kernel < 3 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be an odd integer >= 3")
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    detected = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=max(1.0, min_dist * scale),
        param1=param1,
        param2=param2,
        minRadius=round(min_radius * scale),
        maxRadius=round(max_radius * scale),
    )

    circles: list[Circle] = []
    if detected is not None:
        for x, y, radius in np.round(detected[0]).astype(int):
            circles.append(Circle(x / scale, y / scale, radius / scale))
    circles.sort(key=lambda circle: circle.radius, reverse=True)
    return circles, resized


def annotate(image: np.ndarray, circles: list[Circle], scale: float) -> np.ndarray:
    """在缩放后的图像上绘制圆轮廓、圆心和检测编号。"""
    result = image.copy()
    for index, circle in enumerate(circles, start=1):
        center = (round(circle.x * scale), round(circle.y * scale))
        radius = max(1, round(circle.radius * scale))
        cv2.circle(result, center, radius, (0, 255, 0), 3)
        cv2.circle(result, center, 4, (0, 0, 255), -1)
        cv2.putText(
            result,
            f"{index}: ({circle.x:.1f}, {circle.y:.1f}), r={circle.radius:.1f}",
            (max(0, center[0] - radius), max(20, center[1] - radius - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return result


def process_file(path: Path, output_dir: Path, **parameters: object) -> ImageResult:
    """处理单张图片，同时保存检测结果图和 Canny 边缘图。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {path}")

    started = time.perf_counter()
    circles, resized = detect_circles(image, **parameters)
    elapsed_ms = (time.perf_counter() - started) * 1000
    scale = resized.shape[1] / image.shape[1]
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur_kernel = int(parameters["blur_kernel"])
    param1 = float(parameters["param1"])
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred, param1 / 2, param1)
    annotated = annotate(resized, circles, scale)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / path.name), annotated)
    cv2.imwrite(str(output_dir / f"{path.stem}_edges.png"), edges)
    return ImageResult(path.name, elapsed_ms, circles)


def build_parser() -> argparse.ArgumentParser:
    """创建 OpenCV 基线程序的命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="OpenCV coin localization baseline")
    parser.add_argument("input", type=Path, help="an image file or a directory of images")
    parser.add_argument("-o", "--output", type=Path, default=Path("results/opencv"))
    parser.add_argument("--dp", type=float, default=1.2)
    parser.add_argument("--min-dist", type=float, default=180.0)
    parser.add_argument("--param1", type=float, default=100.0)
    parser.add_argument("--param2", type=float, default=32.0)
    parser.add_argument("--min-radius", type=int, default=120)
    parser.add_argument("--max-radius", type=int, default=400)
    parser.add_argument("--max-dimension", type=int, default=800)
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=11,
        help="odd Gaussian blur kernel; larger values suppress periodic texture",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """批量处理输入图片，并保存 JSON 检测报告。"""
    args = build_parser().parse_args(argv)
    input_path: Path = args.input
    paths = sorted(input_path.glob("*.jpg")) if input_path.is_dir() else [input_path]
    if not paths:
        raise SystemExit(f"no JPG images found in {input_path}")

    parameters = {
        "dp": args.dp,
        "min_dist": args.min_dist,
        "param1": args.param1,
        "param2": args.param2,
        "min_radius": args.min_radius,
        "max_radius": args.max_radius,
        "max_dimension": args.max_dimension,
        "blur_kernel": args.blur_kernel,
    }
    results = [process_file(path, args.output, **parameters) for path in paths]
    report = [asdict(result) for result in results]
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in results:
        print(f"{result.image}: {len(result.circles)} circle(s), {result.elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()

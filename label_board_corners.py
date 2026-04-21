"""Interactive corner-labeling tool for Connect 4 board localization.

This tool iterates over images in data/grid_labels/ and lets you click the four
board corners in this exact order:
    1. upper left
    2. upper right
    3. lower right
    4. lower left

For each labeled image it saves:
    frame_NNNN_corners.npy   -- shape (4, 2), integer pixel coordinates (x, y)

Usage:
    uv run python label_board_corners.py
    uv run python label_board_corners.py --overwrite
    uv run python label_board_corners.py --start-at frame_0042.png

Keyboard shortcuts while labeling:
    p   -- copy the previous image's 4 corners onto the current image
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = Path("data/grid_labels")
WINDOW_NAME = "Label Board Corners"
CORNER_NAMES = [
    "upper left",
    "upper right",
    "lower right",
    "lower left",
]


def annotation_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_corners.npy")


def load_existing_points(image_path: Path) -> list[tuple[int, int]]:
    annotation_path = annotation_path_for(image_path)
    if not annotation_path.exists():
        return []
    points = np.load(annotation_path)
    if points.shape != (4, 2):
        return []
    return [(int(x), int(y)) for x, y in points]


def draw_points(image: np.ndarray, points: list[tuple[int, int]], image_path: Path) -> np.ndarray:
    display = image.copy()

    for idx, point in enumerate(points):
        cv2.circle(display, point, 6, (0, 255, 0), -1)
        cv2.circle(display, point, 10, (0, 0, 0), 2)
        cv2.putText(
            display,
            str(idx + 1),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    if len(points) >= 2:
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(display, start, end, (0, 200, 255), 2)
    if len(points) == 4:
        cv2.line(display, points[-1], points[0], (0, 200, 255), 2)

    instructions = [
        f"Image: {image_path.name}",
        f"Next: {CORNER_NAMES[min(len(points), 3)] if len(points) < 4 else 'press c to confirm'}",
        "Left click: add point",
        "u: undo   r: reset   p: copy previous   c: confirm/save   n: skip   q: quit",
    ]
    y = 28
    for line in instructions:
        cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1)
        y += 28

    return display


def label_image(
    image_path: Path,
    preloaded_points: list[tuple[int, int]] | None = None,
    previous_points: list[tuple[int, int]] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read {image_path}")
        return "skip", []

    points = list(preloaded_points or [])

    def mouse_callback(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(points) >= 4:
            return
        points.append((x, y))
        cv2.imshow(WINDOW_NAME, draw_points(image, points, image_path))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    cv2.imshow(WINDOW_NAME, draw_points(image, points, image_path))

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u"):
            if points:
                points.pop()
                cv2.imshow(WINDOW_NAME, draw_points(image, points, image_path))
        elif key == ord("r"):
            points.clear()
            cv2.imshow(WINDOW_NAME, draw_points(image, points, image_path))
        elif key == ord("p"):
            if previous_points is None or len(previous_points) != 4:
                print(f"No previous 4-corner annotation available for {image_path.name}")
                continue
            points.clear()
            points.extend(previous_points)
            cv2.imshow(WINDOW_NAME, draw_points(image, points, image_path))
        elif key == ord("n"):
            return "skip", points
        elif key == ord("q") or key == 27:
            return "quit", points
        elif key == ord("c"):
            if len(points) != 4:
                print(f"Need 4 points for {image_path.name}; currently have {len(points)}")
                continue
            np.save(annotation_path_for(image_path), np.array(points, dtype=np.int32))
            print(f"Saved {annotation_path_for(image_path)}")
            return "saved", list(points)


def iter_images(image_dir: Path, start_at: str | None) -> list[Path]:
    image_paths = sorted(image_dir.glob("frame_*.png"))
    if start_at is None:
        return image_paths
    for index, path in enumerate(image_paths):
        if path.name == start_at:
            return image_paths[index:]
    raise ValueError(f"Start image not found: {start_at}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(IMAGE_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-at", default=None)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_paths = iter_images(image_dir, args.start_at)

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    saved = 0
    skipped = 0
    previous_points: list[tuple[int, int]] | None = None

    print("Corner order:")
    for index, name in enumerate(CORNER_NAMES, start=1):
        print(f"  {index}. {name}")
    print()

    try:
        for image_path in image_paths:
            existing_points = load_existing_points(image_path)
            if existing_points and not args.overwrite:
                previous_points = existing_points
                skipped += 1
                continue

            result, labeled_points = label_image(
                image_path,
                preloaded_points=existing_points if args.overwrite else None,
                previous_points=previous_points,
            )
            if result == "saved":
                previous_points = labeled_points
                saved += 1
            elif result == "skip":
                skipped += 1
            elif result == "quit":
                break
    finally:
        cv2.destroyAllWindows()

    print(f"Saved: {saved}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
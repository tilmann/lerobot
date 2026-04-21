from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardBBox:
    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width

    def clipped(self, frame_shape: tuple[int, ...]) -> "BoardBBox":
        frame_height, frame_width = frame_shape[:2]
        top = max(0, min(self.top, frame_height - 1))
        left = max(0, min(self.left, frame_width - 1))
        bottom = max(top + 1, min(self.bottom, frame_height))
        right = max(left + 1, min(self.right, frame_width))
        return BoardBBox(top=top, left=left, height=bottom - top, width=right - left)


def _make_kernel(value: int) -> int:
    return max(3, value | 1)


def _smooth_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, kernel_size | 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _find_left_bound(gray: np.ndarray) -> int | None:
    """Find the first strong light-to-dark transition into the full board region."""
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (_make_kernel(width // 70), _make_kernel(height // 70)), 0)
    darkness = 255.0 - blurred.astype(np.float32)
    profile = _smooth_1d(np.percentile(darkness, 70, axis=0), width // 14)
    score = _smooth_1d(np.diff(profile), width // 45)

    start = max(2, int(width * 0.15))
    end = max(start + 1, int(width * 0.75))
    window = score[start:end]
    if window.size == 0:
        return None

    peak_offset = int(np.argmax(window))
    peak_index = start + peak_offset
    peak_value = float(score[peak_index])
    baseline = float(np.median(window))
    threshold = baseline + max(1.5, 0.28 * (peak_value - baseline))

    for index in range(start, peak_index + 1):
        if score[index] >= threshold:
            return index
    return peak_index


def _find_right_bound(gray: np.ndarray, left_bound: int) -> int:
    """Find the first strong dark-to-light transition after the board body."""
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (_make_kernel(width // 70), _make_kernel(height // 70)), 0)
    darkness = 255.0 - blurred.astype(np.float32)
    profile = _smooth_1d(np.percentile(darkness, 70, axis=0), width // 14)
    score = _smooth_1d(-np.diff(profile), width // 45)

    min_offset = int(width * 0.18)
    start = min(width - 2, left_bound + min_offset)
    end = max(start + 1, min(width - 1, left_bound + int(width * 0.5)))
    window = score[start:end]
    if window.size == 0:
        return min(width - 1, left_bound + int(width * 0.38))

    peak_offset = int(np.argmax(window))
    peak_index = start + peak_offset
    peak_value = float(score[peak_index])
    baseline = float(np.median(window))
    threshold = baseline + max(1.0, 0.25 * (peak_value - baseline))

    for index in range(start, peak_index + 1):
        if score[index] >= threshold:
            return index
    return peak_index


def _find_vertical_span(gray: np.ndarray, left: int, right: int) -> tuple[int, int]:
    """Estimate top and bottom from darkness density inside the board strip."""
    height, width = gray.shape
    strip = gray[:, max(0, left):min(width, right)]
    if strip.size == 0:
        return 0, height - 1

    blurred = cv2.GaussianBlur(strip, (_make_kernel(strip.shape[1] // 20), _make_kernel(height // 70)), 0)
    darkness = 255.0 - blurred.astype(np.float32)
    row_score = _smooth_1d(np.mean(darkness, axis=1), height // 50)

    high = float(np.percentile(row_score, 92))
    low = float(np.percentile(row_score, 35))
    threshold = low + 0.4 * (high - low)
    active_rows = np.where(row_score >= threshold)[0]
    if active_rows.size == 0:
        return 0, height - 1

    top = int(active_rows[0])
    bottom = int(active_rows[-1])
    return top, bottom


def detect_board_bbox(
    frame: np.ndarray,
    *,
    margin_ratio: float = 0.08,
    min_area_ratio: float = 0.025,
) -> BoardBBox | None:
    """Detect the Connect 4 board from its first strong left light-to-dark edge."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape (H, W, 3).")

    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left_bound = _find_left_bound(gray)
    if left_bound is None:
        return None

    right_bound = _find_right_bound(gray, left_bound)
    top, bottom = _find_vertical_span(gray, left_bound, right_bound)

    width = max(int(frame_width * min_area_ratio), right_bound - left_bound)
    height = max(int(frame_height * 0.25), bottom - top)
    margin_x = int(width * margin_ratio)
    margin_top = int(height * (margin_ratio + 0.03))
    margin_bottom = int(height * (margin_ratio + 0.12))

    bbox = BoardBBox(
        top=top - margin_top,
        left=left_bound - margin_x,
        height=height + margin_top + margin_bottom,
        width=width + 2 * margin_x,
    ).clipped(frame.shape)
    return bbox


BOARD_CROP_W = 224
BOARD_CROP_H = 192  # 7:6 aspect ratio

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel of an LAB image to normalise contrast."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


_CORNER_NAMES = ["upper left", "upper right", "lower right", "lower left"]


def corners_to_warp_crop(
    frame: np.ndarray,
    corners: np.ndarray,
    out_w: int = BOARD_CROP_W,
    out_h: int = BOARD_CROP_H,
) -> np.ndarray:
    """Perspective-warp frame using 4 board corners to a normalised rectangle.

    corners: (4, 2) array, order UL, UR, LR, LL (x, y)
    """
    src = corners.astype(np.float32)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (out_w, out_h))
    return apply_clahe(warped)


def run_corner_calibration(
    frame: np.ndarray,
    window_name: str = "Calibrate Board",
) -> np.ndarray | None:
    """Interactive 4-corner selection on a single frame.

    Returns (4, 2) int32 corners in order UL, UR, LR, LL (x, y), or None if cancelled.
    """
    points: list[tuple[int, int]] = []

    def _draw(img: np.ndarray) -> np.ndarray:
        display = img.copy()
        for idx, pt in enumerate(points):
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
            cv2.circle(display, pt, 10, (0, 0, 0), 2)
            cv2.putText(
                display, str(idx + 1), (pt[0] + 8, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
        if len(points) >= 2:
            for a, b in zip(points[:-1], points[1:]):
                cv2.line(display, a, b, (0, 200, 255), 2)
        if len(points) == 4:
            cv2.line(display, points[-1], points[0], (0, 200, 255), 2)
        next_label = _CORNER_NAMES[min(len(points), 3)] if len(points) < 4 else "done"
        for row_idx, line in enumerate([
            f"Click: {next_label}",
            "u: undo  r: reset  c: confirm  ESC: cancel",
        ]):
            y = 28 + row_idx * 28
            cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1)
        return display

    def _mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.imshow(window_name, _draw(frame))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse)
    cv2.imshow(window_name, _draw(frame))

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and points:
            points.pop()
            cv2.imshow(window_name, _draw(frame))
        elif key == ord("r"):
            points.clear()
            cv2.imshow(window_name, _draw(frame))
        elif key == ord("c"):
            if len(points) != 4:
                continue
            cv2.destroyWindow(window_name)
            return np.array(points, dtype=np.int32)
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None


def crop_board(frame: np.ndarray) -> tuple[np.ndarray, BoardBBox | None]:
    bbox = detect_board_bbox(frame)
    if bbox is None:
        return frame.copy(), None
    return frame[bbox.top:bbox.bottom, bbox.left:bbox.right].copy(), bbox


def draw_board_bbox(frame: np.ndarray, bbox: BoardBBox | None) -> np.ndarray:
    display = frame.copy()
    if bbox is None:
        return display
    cv2.rectangle(display, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 255, 0), 2)
    cv2.putText(
        display,
        "board crop",
        (bbox.left, max(18, bbox.top - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    return display
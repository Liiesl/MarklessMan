"""Bare-ONNX CPU detection for MarklessMan (no ultralytics at inference time).

Loads weights/best.onnx with onnxruntime CPUExecutionProvider only and runs:
    letterbox(640) -> session -> conf filter -> NMS -> unscale to pixels.

Exported graph is raw (end2end=False): output0 is (1, 5, 8400) = cxcywh in
640-letterbox space + sigmoided single-class score. NMS happens here.

Output matches inference.detect_yolo format: [(x, y, w, h), ...] int boxes.

Usage:
    py -3.13 detect_onnx.py --images test/bench/images --out test/onnx_det --limit 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMGSZ = 640
CONF_DEFAULT = 0.25
IOU_DEFAULT = 0.5
PAD_VALUE = 114


def letterbox(img_bgr: np.ndarray, size: int = IMGSZ):
    """Resize+pad to size x size. Returns (tensor CHW float32 RGB, scale, (pad_x, pad_y))."""
    h, w = img_bgr.shape[:2]
    s = min(size / h, size / w)
    nh, nw = round(h * s), round(w * s)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = size - nh, size - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    t = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(t), s, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS over xyxy boxes (pure numpy, no torch). Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    x0, y0, x1, y1 = boxes.T
    area = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        ovr = inter / np.maximum(area[i] + area[order[1:]] - inter, 1e-9)
        order = order[1:][ovr <= iou]
    return np.array(keep, dtype=np.int64)


class OnnxDetector:
    """Bare onnxruntime CPU detector. Session is created once per weights path."""

    def __init__(self, weights: str | Path = "weights/best.onnx",
                 conf: float = CONF_DEFAULT, iou: float = IOU_DEFAULT,
                 imgsz: int = IMGSZ):
        import onnxruntime as ort
        self.weights = str(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.sess = ort.InferenceSession(self.weights, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def raw(self, img_bgr: np.ndarray):
        """Run session. Returns (xyxy boxes in orig pixels float32, scores)."""
        h, w = img_bgr.shape[:2]
        t, s, (px, py) = letterbox(img_bgr, self.imgsz)
        out = self.sess.run(None, {self.in_name: t})[0][0]  # (C, 8400)
        if out.shape[0] > 5:  # multi-class fallback: best class wins
            cls = out[4:].argmax(axis=0).astype(np.float32)
            scr = out[4:].max(axis=0)
            pred = np.stack([out[0], out[1], out[2], out[3], scr, cls], axis=1)
        else:  # single class: (cx, cy, w, h, score)
            pred = np.concatenate([out[:5].T, np.zeros((out.shape[1], 1), np.float32)], axis=1)
        pred = pred[pred[:, 4] >= self.conf]
        if len(pred) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        xyxy = np.empty((len(pred), 4), np.float32)
        xyxy[:, 0] = (pred[:, 0] - pred[:, 2] / 2 - px) / s  # x0
        xyxy[:, 1] = (pred[:, 1] - pred[:, 3] / 2 - py) / s  # y0
        xyxy[:, 2] = (pred[:, 0] + pred[:, 2] / 2 - px) / s  # x1
        xyxy[:, 3] = (pred[:, 1] + pred[:, 3] / 2 - py) / s  # y1
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
        keep = _nms(xyxy, pred[:, 4].astype(np.float32), self.iou)
        return xyxy[keep], pred[keep, 4]

    def detect(self, img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Boxes as (x, y, w, h) ints — same format as inference.detect_yolo."""
        boxes, _ = self.raw(img_bgr)
        return [(int(x0), int(y0), int(max(0, x1 - x0)), int(max(0, y1 - y0)))
                for x0, y0, x1, y1 in boxes]


_cache: dict[str, OnnxDetector] = {}


def detect_onnx(img_bgr: np.ndarray, weights: str | Path = "weights/best.onnx",
                conf: float = CONF_DEFAULT, iou: float = IOU_DEFAULT) -> list[tuple[int, int, int, int]]:
    """Cached bare-ORT detect. Mirrors inference.detect_yolo signature."""
    key = str(weights)
    det = _cache.get(key)
    if det is None or det.conf != conf or det.iou != iou:
        det = OnnxDetector(weights, conf=conf, iou=iou)
        _cache[key] = det
    return det.detect(img_bgr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bare-ONNX CPU detect + draw boxes.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights/best.onnx")
    ap.add_argument("--conf", type=float, default=CONF_DEFAULT)
    ap.add_argument("--iou", type=float, default=IOU_DEFAULT)
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    from tqdm import tqdm

    det = OnnxDetector(args.weights, conf=args.conf, iou=args.iou)
    src = sorted(Path(args.images).glob("*.jpg"))[: args.limit]
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    for p in tqdm(src, desc="onnx", unit="img"):
        img = cv2.imread(str(p))
        boxes, scores = det.raw(img)
        total += len(boxes)
        vis = img.copy()
        for (x0, y0, x1, y1), s in zip(boxes, scores):
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 3)
            cv2.putText(vis, f"{s:.2f}", (int(x0), max(0, int(y0) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(str(dst / f"{p.stem}_boxed.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"done: {len(src)} images, {total} boxes -> {dst}")


if __name__ == "__main__":
    main()

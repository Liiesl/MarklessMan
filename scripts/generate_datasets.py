"""Unified synthetic dataset generator for MarklessMan 2-stage pipeline.

Two INDEPENDENT outputs from one shared page plan (same seed/split):

det (YOLO, single class `watermark`):
    <out-det>/images/{train,val}/ + labels/{train,val}/ + data.yaml
    full clean page -> paste 1-3 NON-OVERLAPPING watermarks -> JPG + YOLO txt
clean (SLBR finetune, native CLWD layout, native 256 tiles):
    <out-clean>/train/{Watermarked_image,Watermark_free_image,Mask,Alpha,Watermark}/
    <out-clean>/test/{...}/   (val split is named `test`, exactly what
    CLWDDataset expects: <root>/train/ + <root>/test/, ids via listdir)
    for each pasted box: padded rect (tight box + PAD context) divided into
    NATIVE 256 tiles (stride STRIDE, same as Rust inference) -> no resize

Watermark source: pre-generated synthetic bank + real Shimu bank, 50-50 GLOBAL
over placed boxes (not per page). Both banks go through the IDENTICAL geometry
pipeline (scale/rotate/anchor/jitter/JPEG), so synth matches real placement.

Compositing uses NATIVE PNG alpha (no extra alpha scaling):
    J = A * W_rgb + (1 - A) * I,  A = wm_alpha / 255
Scale is biased toward native pixel size: s ~ N(1.0, 0.15), clamped,
shrink-to-fit only, min long side MIN_WM_LONG_SIDE.

Multiprocessed over pages (--workers). Deterministic for a given
(seed, page_idx): counts/sources come from a pre-assigned global plan
(seed:count:i / seed:sources shuffle), geometry from a per-page RNG, so
output is identical regardless of worker count or scheduling.

Usage (dry run, dummy cleans):
    py -3.13 generate_datasets.py --out ./datasets/dryrun100 --num-pages 100 --dummy-cleans --seed 7

Usage (full comfortable gen, 2071 cleans x 5 split-safe variants):
    py -3.13 generate_datasets.py --out-det ./datasets/det --out-clean ./datasets/clean \\
        --wm-root "./Shimu WMs" --synth-root ./synth_bank --seed 7 --variants-per-page 5

Outputs:
    <out-det>/images/{train,val}/ + labels/{train,val}/ + data.yaml + preview/ + stats.json
    <out-clean>/train/... + test/... + preview/ + stats.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

SLBR_SIZE = 256
YOLO_CLASS = 0
MIN_WM_LONG_SIDE = 64  # upscale small logos so long side >= 64px
OVERLAP_ATTEMPTS = 25
PAD = 64     # context padding around each tight box (mirrors Rust inference --pad)
STRIDE = 192  # tile stride, 256 -> 64px overlap (mirrors Rust inference --stride)

CLWD_SUBS = ("Watermarked_image", "Watermark_free_image", "Mask", "Alpha", "Watermark")
CLWD_EXT = {"Watermarked_image": ".jpg", "Watermark_free_image": ".jpg",
            "Mask": ".png", "Alpha": ".png", "Watermark": ".png"}

ANCHORS = ["top-left", "top-center", "top-right", "center",
           "bottom-left", "bottom-center", "bottom-right", "random"]


# ---------------------------------------------------------------- native tiling (mirrors Rust inference:
# clean-dml-cli padded_bbox + tile_origins, and test/tile_vs_resize method_B).
# Training tiles are cut at NATIVE scale, no resize — exactly what the model
# sees at inference (one 256 forward per tile, feather-mosaicked).
def padded_bbox(x, y, w, h, img_w, img_h, pad):
    """Tight box + surrounding context, clamped to the image -> (x, y, w, h)."""
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img_w, x + w + pad), min(img_h, y + h + pad)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def tile_origins(reg0, reg_len, img_len, size, stride):
    """Tile origins (1-D) covering [reg0, reg0+reg_len), tiles kept inside image."""
    if reg_len <= size:
        o = reg0 - (size - reg_len) // 2
        return [int(min(max(o, 0), max(0, img_len - size)))]
    first = max(0, min(reg0, img_len - size))
    last = max(first, min(reg0 + reg_len - size, img_len - size))
    origins = list(range(first, last, stride))
    if origins and last - origins[-1] <= 64:
        origins[-1] = last  # snap: avoid near-duplicate tile at the clamp
    elif not origins or origins[-1] != last:
        origins.append(last)
    return origins


# ---------------------------------------------------------------- clean pages
def make_dummy_clean(w: int, h: int, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    """Manga-ish placeholder: paper white + panels + bubbles + screentone noise."""
    base = np.ones((h, w, 3), dtype=np.uint8) * int(rng.uniform(235, 255))
    img = Image.fromarray(base, "RGB")
    d = ImageDraw.Draw(img)
    # panels: 2-4 stacked rectangles with black gutters
    n_panels = rng.randint(2, 4)
    gutter = int(w * 0.02)
    y0 = 0
    for i in range(n_panels):
        ph = h // n_panels
        x0, y1 = gutter, min(y0 + ph - gutter, h - 2)
        shade = int(rng.uniform(180, 245))
        d.rectangle([gutter, y0 + gutter // 2, w - gutter, y1], fill=(shade, shade, shade),
                    outline=(20, 20, 20), width=max(2, w // 200))
        # diagonal speedlines in some panels
        if rng.random() < 0.6:
            for _ in range(rng.randint(6, 18)):
                x = rng.randint(gutter, w - gutter)
                d.line([(x, y0), (x + rng.randint(-40, 40), y1)], fill=(60, 60, 60), width=1)
        # speech bubble
        if rng.random() < 0.8:
            bw, bh = rng.randint(w // 5, w // 2), rng.randint(h // 40, h // 16)
            bx, by = rng.randint(gutter, w - gutter - bw), rng.randint(y0, max(y0, y1 - bh))
            d.ellipse([bx, by, bx + bw, by + bh], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            # fake text lines
            for li in range(rng.randint(1, 3)):
                ly = by + 6 + li * (bh // 4)
                d.line([(bx + 8, ly), (bx + bw - 8, ly)], fill=(30, 30, 30), width=1)
        y0 += ph
    # screentone grain
    arr = np.asarray(img).astype(np.int16)
    grain = np_rng.integers(-12, 13, size=arr.shape)
    arr = np.clip(arr + grain * 0.6, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------- helpers
def load_real_bank(wm_root: Path) -> list[Path]:
    pngs = sorted([p for p in wm_root.rglob("*.png") if "Presets" not in p.parts])
    if not pngs:
        raise SystemExit(f"No real watermark PNGs under {wm_root}")
    return pngs


def load_synth_bank(synth_root: Path | None) -> list[Path]:
    """Pre-generated synthetic bank (synthetic_wm.py --bank). Empty list = absent."""
    if synth_root is None or not Path(synth_root).exists():
        return []
    pngs = sorted([p for p in Path(synth_root).rglob("*.png")
                   if "preview" not in p.parts and "Presets" not in p.parts])
    return pngs


def collect_cleans(clean_dir: Path | None) -> list[Path]:
    if clean_dir is None:
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([p for p in clean_dir.rglob("*") if p.suffix.lower() in exts])
    return files


def deal_synth_picks(sources_per_page: list[list[str]], synth_len: int,
                     seed: int) -> tuple[list[list[tuple]], bool]:
    """Align each synth slot with a bank index.

    Without replacement when the bank covers all synth slots (every placed
    synthetic watermark file-unique); with-replacement cycling + warning
    otherwise. Real slots always get None (sampled with replacement).
    """
    need = sum(1 for pg in sources_per_page for s in pg if s == "synth")
    if synth_len <= 0:
        return [[(s, None) for s in pg] for pg in sources_per_page], False
    order = list(range(synth_len))
    random.Random(f"{seed}:synth-order").shuffle(order)
    exact = synth_len >= need
    if not exact:
        print(f"[warn] synth bank shortfall: need {need} unique, have {synth_len} "
              f"-> reusing with replacement")
        order = (order * ((need + synth_len - 1) // synth_len))[:need]
    it = iter(order)
    picks = [[(s, next(it) if s == "synth" else None) for s in pg]
             for pg in sources_per_page]
    return picks, exact


def plan_sources(n_pages: int, seed: int) -> list[list[str]]:
    """Per-page watermark-source slots with exact 50-50 global real:synth.

    Counts (1-3/page) come from an independent `seed:count:i` stream so the
    plan is fixed before workers start; the source labels are shuffled with
    `seed:sources` and dealt out in plan order. Workers loop over their slot
    list (no per-page randint), so the global ratio holds regardless of
    worker count/scheduling. Skipped overlaps consume their slot, so the
    final placed ratio is exact up to skips (reported in stats).
    """
    counts = [random.Random(f"{seed}:count:{i}").randint(1, 3) for i in range(n_pages)]
    total = sum(counts)
    n_synth = total // 2
    labels = ["real"] * (total - n_synth) + ["synth"] * n_synth
    r = random.Random(f"{seed}:sources")
    r.shuffle(labels)
    out: list[list[str]] = []
    it = iter(labels)
    for c in counts:
        out.append([next(it) for _ in range(c)])
    return out


def anchor_position(anchor: str, pg_w: int, pg_h: int, wm_w: int, wm_h: int,
                    rng: random.Random, margin: int = 8) -> tuple[int, int]:
    if anchor == "random":
        x = rng.randint(0, max(0, pg_w - wm_w))
        y = rng.randint(0, max(0, pg_h - wm_h))
        return x, y
    xs = {"top-left": margin, "center": (pg_w - wm_w) // 2, "top-center": (pg_w - wm_w) // 2,
          "bottom-center": (pg_w - wm_w) // 2, "top-right": pg_w - wm_w - margin,
          "bottom-left": margin, "bottom-right": pg_w - wm_w - margin}
    ys = {"top-left": margin, "top-center": margin, "top-right": margin,
          "center": (pg_h - wm_h) // 2, "bottom-left": pg_h - wm_h - margin,
          "bottom-center": pg_h - wm_h - margin, "bottom-right": pg_h - wm_h - margin}
    x = int(np.clip(xs.get(anchor, 0) + rng.randint(-10, 10), 0, max(0, pg_w - wm_w)))
    y = int(np.clip(ys.get(anchor, 0) + rng.randint(-10, 10), 0, max(0, pg_h - wm_h)))
    return x, y


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def overlay_rgba(page: Image.Image, wm: Image.Image, x: int, y: int,
                 ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Paste wm onto page with NATIVE PNG alpha. Returns (composited, alpha_full, wrgb_full)."""
    pg = np.asarray(page).astype(np.float32)
    wm_arr = np.asarray(wm).astype(np.float32)  # HxWx4 expected
    if wm_arr.shape[2] == 3:  # treat RGB-only files as opaque
        opaque = np.full(wm_arr.shape[:2], 255.0, dtype=np.float32)
        wm_arr = np.dstack([wm_arr, opaque])
    wh, ww = wm_arr.shape[:2]
    H, W = pg.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + ww), min(H, y + wh)
    if x1 <= x0 or y1 <= y0:
        return page, np.zeros((H, W), np.float32), np.zeros((H, W, 3), np.float32)
    wx0, wy0 = x0 - x, y0 - y
    patch = wm_arr[wy0:wy0 + (y1 - y0), wx0:wx0 + (x1 - x0)]
    a = patch[..., 3:4] / 255.0  # native PNG alpha, no extra scaling
    rgb = patch[..., :3]
    pg[y0:y1, x0:x1] = a * rgb + (1.0 - a) * pg[y0:y1, x0:x1]
    alpha_full = np.zeros((H, W), np.float32)
    alpha_full[y0:y1, x0:x1] = a[..., 0]
    wrgb_full = np.zeros((H, W, 3), np.float32)
    wrgb_full[y0:y1, x0:x1] = rgb
    return Image.fromarray(np.clip(pg, 0, 255).astype(np.uint8)), alpha_full, wrgb_full


def jpeg_bytes(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------- per-page worker
def process_page(page_idx: int, clean_path, split: str, ctx: dict) -> dict:
    """Generate one det page + its CLWD crops. All randomness from per-page RNGs."""
    rng = random.Random(f"{ctx['seed']}:{page_idx}")
    np_rng = np.random.default_rng([ctx["seed"], page_idx])
    real_bank, synth_bank = ctx["real_bank"], ctx["synth_bank"]
    out_det, out_clean = ctx["out_det"], ctx["out_clean"]
    S = ctx["slbr_size"]
    sources = ctx["picks_per_page"][page_idx]  # list of (src, bank_idx|None)

    # --- clean background
    if ctx["use_dummy"]:
        pw, ph = rng.choice([(720, 1280), (760, 1400), (800, 1600), (720, 1800)])
        clean = make_dummy_clean(pw, ph, rng, np_rng)
    else:
        cp = clean_path if clean_path is not None else rng.choice(ctx["cleans"])
        clean = Image.open(cp).convert("RGB")
        if clean.width > 900:  # cap memory, keep aspect
            clean = clean.resize((900, int(clean.height * 900 / clean.width)), Image.LANCZOS)
    pg_w, pg_h = clean.size
    page = clean.copy()
    alpha_accum = np.zeros((pg_h, pg_w), np.float32)
    wrgb_accum = np.zeros((pg_h, pg_w, 3), np.float32)
    boxes: list[tuple[int, int, int, int]] = []
    box_src: list[str] = []
    scales: list[float] = []
    tiny = huge = skipped = fallback_real = reuse_synth = 0

    # --- paste NON-OVERLAPPING watermarks. `sources` fixes the global 50-50;
    # geometry below is IDENTICAL for real and synth (scale/rotate/anchor/JPEG).
    # Synth slots carry a pre-dealt bank index (file-unique when the bank
    # covers all slots); real slots sample with replacement.
    for src, bidx in sources:
        eff_src = src
        if src == "synth" and synth_bank and bidx is not None and 0 <= bidx < len(synth_bank):
            wm_path = synth_bank[bidx]
        else:
            bank = real_bank if (src == "real" or not synth_bank) else synth_bank
            if src == "synth" and bank is real_bank:
                eff_src = "real"
                fallback_real += 1
            elif src == "synth":
                reuse_synth += 1  # bank short: with-replacement reuse
            wm_path = bank[rng.randrange(len(bank))]
        try:
            wm0 = Image.open(wm_path).convert("RGBA")
        except Exception:
            continue
        # scale biased toward native pixel size; shrink-to-fit only
        s = min(max(rng.gauss(1.0, 0.15), 0.7), 1.4)
        nw, nh = int(wm0.width * s), int(wm0.height * s)
        if nw > pg_w:
            k = pg_w / max(1, nw)
            nw, nh = max(8, int(nw * k)), max(8, int(nh * k))
        if nh > pg_h * 0.9:
            k = (pg_h * 0.9) / max(1, nh)
            nw, nh = max(8, int(nw * k)), max(8, int(nh * k))
        if max(nw, nh) < MIN_WM_LONG_SIDE:  # min size, keep aspect
            k = MIN_WM_LONG_SIDE / max(1, max(nw, nh))
            nw, nh = int(nw * k), int(nh * k)
        scales.append(nw / max(1, wm0.width))
        wm = wm0.resize((nw, nh), Image.LANCZOS)
        if rng.random() < 0.3:
            wm = wm.rotate(rng.uniform(-8, 8), expand=True, resample=Image.BICUBIC)
        # tight box from watermark's own alpha (excludes rotation padding)
        wm_alpha = np.asarray(wm)[..., 3]
        ay, ax = np.where(wm_alpha > 5)
        if len(ax) == 0:
            continue
        tx0, ty0 = int(ax.min()), int(ay.min())
        tw = int(ax.max()) + 1 - tx0
        th = int(ay.max()) + 1 - ty0
        placed = None
        for _attempt in range(OVERLAP_ATTEMPTS):
            anchor = ANCHORS[rng.randrange(len(ANCHORS))] if rng.random() < 0.7 else "random"
            x, y = anchor_position(anchor, pg_w, pg_h, wm.width, wm.height, rng)
            cand = (x + tx0, y + ty0, tw, th)
            if not any(boxes_overlap(cand, b) for b in boxes):
                placed = (x, y, cand)
                break
        if placed is None:
            skipped += 1
            continue
        x, y, (bx, by, bw, bh) = placed
        page, a_full, w_full = overlay_rgba(page, wm, x, y)
        boxes.append((bx, by, bw, bh))
        box_src.append(eff_src)
        alpha_accum = np.maximum(alpha_accum, a_full)
        m = a_full[..., None] > 0.02
        wrgb_accum = np.where(m, w_full, wrgb_accum)
        tiny += int(min(bw, bh) < 32)
        huge += int(max(bw / pg_w, bh / pg_h) > 0.6)

    # JPEG recompression like real uploads
    page = jpeg_bytes(page, rng.randint(75, 95))

    # --- det output: YOLO page + label (train/val)
    stem = f"page_{page_idx:05d}"
    page.save(out_det / "images" / split / f"{stem}.jpg", quality=92)
    with open(out_det / "labels" / split / f"{stem}.txt", "w") as f:
        for (bx, by, bw, bh) in boxes:
            xc = (bx + bw / 2) / pg_w
            yc = (by + bh / 2) / pg_h
            # clip to [0,1]
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            wn, hn = min(bw / pg_w, 1.0), min(bh / pg_h, 1.0)
            f.write(f"{YOLO_CLASS} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    # --- clean output: native CLWD layout (val -> test). Per tight box, tile
    # the padded rect at NATIVE scale (no resize) — same tiles inference runs.
    clwd_split = "train" if split == "train" else "test"
    pad, stride = ctx["pad"], ctx["stride"]
    page_np = np.asarray(page)
    clean_np = np.asarray(clean)
    alpha_u8 = (np.clip(alpha_accum, 0, 1) * 255).astype(np.uint8)
    mask_u8 = (alpha_accum > 0.1).astype(np.uint8) * 255
    wrgb_u8 = np.clip(wrgb_accum, 0, 255).astype(np.uint8)
    cids: list[str] = []
    cid_src: list[str] = []
    tiles_per_box: list[int] = []
    skipped_tiles = 0
    for k, (bx, by, bw, bh) in enumerate(boxes):
        rx, ry, rw, rh = padded_bbox(bx, by, bw, bh, pg_w, pg_h, pad)
        xs = tile_origins(rx, rw, pg_w, S, stride)
        ys = tile_origins(ry, rh, pg_h, S, stride)
        n_tiles = 0
        for t, (ox, oy) in enumerate([(x, y) for y in ys for x in xs]):
            if ox < 0 or oy < 0 or ox + S > pg_w or oy + S > pg_h:
                skipped_tiles += 1  # same guard as inference (page edge)
                continue
            j = page_np[oy:oy + S, ox:ox + S]
            ref = clean_np[oy:oy + S, ox:ox + S]
            ma = alpha_u8[oy:oy + S, ox:ox + S]
            mb = mask_u8[oy:oy + S, ox:ox + S]
            wrgb = wrgb_u8[oy:oy + S, ox:ox + S]
            cid = f"{page_idx:05d}_{k}_t{t}"  # page-based: deterministic under threading
            cv2.imwrite(str(out_clean / clwd_split / "Watermarked_image" / f"{cid}.jpg"),
                        cv2.cvtColor(j, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_clean / clwd_split / "Watermark_free_image" / f"{cid}.jpg"),
                        cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_clean / clwd_split / "Mask" / f"{cid}.png"), mb)
            cv2.imwrite(str(out_clean / clwd_split / "Alpha" / f"{cid}.png"), ma)
            cv2.imwrite(str(out_clean / clwd_split / "Watermark" / f"{cid}.png"),
                        cv2.cvtColor(wrgb, cv2.COLOR_RGB2BGR))
            cids.append(cid)
            cid_src.append(box_src[k])
            n_tiles += 1
        tiles_per_box.append(n_tiles)

    # --- preview for first 12 pages: GT box (green) + padded rect (yellow)
    # + native tile grid (cyan), mirroring inference tiling
    if page_idx < 12:
        pv = page.copy()
        dr = ImageDraw.Draw(pv)
        for (bx, by, bw, bh) in boxes:
            dr.rectangle([bx, by, bx + bw, by + bh], outline=(0, 255, 0), width=3)
            rx, ry, rw, rh = padded_bbox(bx, by, bw, bh, pg_w, pg_h, pad)
            dr.rectangle([rx, ry, rx + rw, ry + rh], outline=(255, 220, 0), width=2)
            for ox in tile_origins(rx, rw, pg_w, S, stride):
                for oy in tile_origins(ry, rh, pg_h, S, stride):
                    dr.rectangle([ox, oy, ox + S, oy + S], outline=(0, 255, 255), width=1)
        pv.thumbnail((512, 1024))
        pv.save(out_det / "preview" / f"{stem}_{split}.jpg", quality=88)

    return {"page_idx": page_idx, "split": split, "clwd_split": clwd_split,
            "n_boxes": len(boxes), "cids": cids, "cid_src": cid_src,
            "box_src": box_src, "scales": scales, "tiny": tiny, "huge": huge,
            "skipped": skipped, "tiles_per_box": tiles_per_box,
            "skipped_tiles": skipped_tiles, "fallback_real": fallback_real,
            "reuse_synth": reuse_synth}


# ---------------------------------------------------------------- main gen
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-root", default=r"E:\vibecode project\MarklessMan\Shimu WMs")
    ap.add_argument("--synth-root", default=r"E:\vibecode project\MarklessMan\synth_bank",
                    help="Pre-generated synthetic watermark bank (synthetic_wm.py --bank). "
                         "Missing dir -> real-only fallback with warning.")
    ap.add_argument("--clean-dir", default="datasets/clean")
    ap.add_argument("--out", default=None,
                    help="Legacy single prefix: det=<out>/det, clean=<out>/clean. "
                         "Ignored when --out-det/--out-clean are set.")
    ap.add_argument("--out-det", default=None)
    ap.add_argument("--out-clean", default=None)
    ap.add_argument("--num-pages", type=int, default=100)
    ap.add_argument("--dummy-cleans", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--variants-per-page", type=int, default=1,
                    help="N seeded variants per unique clean (split-safe per clean). "
                         "Overrides --num-pages when >1 with real cleans.")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--slbr-size", type=int, default=SLBR_SIZE)
    ap.add_argument("--pad", type=int, default=PAD,
                    help="Context padding around each tight box; must match Rust inference --pad.")
    ap.add_argument("--stride", type=int, default=STRIDE,
                    help="Tile stride; must match Rust inference --stride.")
    args = ap.parse_args()

    if args.out_det and args.out_clean:
        out_det, out_clean = Path(args.out_det), Path(args.out_clean)
    elif args.out:
        out_det, out_clean = Path(args.out) / "det", Path(args.out) / "clean"
    else:
        out_det = Path(args.out_det) if args.out_det else Path("datasets/det")
        out_clean = Path(args.out_clean) if args.out_clean else Path("datasets/clean")

    rng = random.Random(args.seed)
    real_bank = load_real_bank(Path(args.wm_root))
    synth_bank = load_synth_bank(Path(args.synth_root) if args.synth_root else None)
    if not synth_bank:
        print(f"[warn] no synthetic PNGs under {args.synth_root} -> real-only fallback "
              f"(50-50 disabled until the pre-generated bank is present)")

    cleans = collect_cleans(Path(args.clean_dir)) if args.clean_dir else []
    use_dummy = args.dummy_cleans or not cleans
    if args.clean_dir and not cleans and not args.dummy_cleans:
        print(f"[info] no clean images under {args.clean_dir} -> using dummy placeholders")
    if not use_dummy and args.variants_per_page <= 1 and len(cleans) < args.num_pages:
        print(f"[warn] only {len(cleans)} clean files < {args.num_pages} pages; will reuse with replacement")

    # prepare dirs: det uses train/val, clean uses native CLWD train/test
    for split in ("train", "val"):
        (out_det / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_det / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for sub in CLWD_SUBS:
            (out_clean / split / sub).mkdir(parents=True, exist_ok=True)
    (out_det / "preview").mkdir(parents=True, exist_ok=True)
    (out_clean / "preview").mkdir(parents=True, exist_ok=True)

    # build per-page plan: (clean_path|None, split); split-safe per clean when using variants
    if not use_dummy and args.variants_per_page > 1 and cleans:
        V = args.variants_per_page
        order_c = cleans.copy()
        rng.shuffle(order_c)
        n_val_c = int(round(len(order_c) * args.val_ratio))
        val_set = set(order_c[:n_val_c])
        plan: list[tuple] = []
        for cp in order_c:
            split = "val" if cp in val_set else "train"
            for _ in range(V):
                plan.append((cp, split))
        rng.shuffle(plan)
        print(f"[info] {len(order_c)} unique cleans x {V} variants = {len(plan)} pages (split-safe per clean)")
    else:
        n_v = int(round(args.num_pages * args.val_ratio))
        plan = [(None, ("val" if i >= args.num_pages - n_v else "train")) for i in range(args.num_pages)]

    sources_per_page = plan_sources(len(plan), args.seed)

    # --- pre-generated synth bank, dealt without replacement when it covers
    # all synth slots (every placed synthetic watermark file-unique).
    picks_per_page, synth_exact = deal_synth_picks(sources_per_page, len(synth_bank), args.seed)

    ctx = {"real_bank": real_bank, "synth_bank": synth_bank, "cleans": cleans,
           "out_det": out_det, "out_clean": out_clean, "slbr_size": args.slbr_size,
           "pad": args.pad, "stride": args.stride,
           "use_dummy": use_dummy, "seed": args.seed,
           "picks_per_page": picks_per_page}
    print(f"[info] {len(plan)} pages on {args.workers} workers "
          f"(real={len(real_bank)} synth={len(synth_bank)}[pre-generated] "
          f"unique={synth_exact})")

    results: list[dict] = []
    # NOTE: process pool, not thread pool: sidesteps the GIL. ctx must stay picklable.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_page, i, cp, sp, ctx): i for i, (cp, sp) in enumerate(plan)}
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(plan), desc="pages", unit="page"):
            results.append(fut.result())
    results.sort(key=lambda r: r["page_idx"])

    (out_det / "data.yaml").write_text(
        f"path: {out_det.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: watermark\n", encoding="utf-8")

    scales = [s for r in results for s in r["scales"]]
    box_src_all = [s for r in results for s in r["box_src"]]
    n_real = sum(1 for s in box_src_all if s == "real")
    n_synth = sum(1 for s in box_src_all if s == "synth")
    tile_src_all = [s for r in results for s in r["cid_src"]]
    n_real_tiles = sum(1 for s in tile_src_all if s == "real")
    n_synth_tiles = sum(1 for s in tile_src_all if s == "synth")
    tpb = [t for r in results for t in r["tiles_per_box"]]
    n_train_pages = sum(1 for _, s in plan if s == "train")
    n_boxes = sum(r["n_boxes"] for r in results)
    tr_crops = sum(len(r["cids"]) for r in results if r["split"] == "train")
    te_crops = sum(len(r["cids"]) for r in results if r["split"] == "val")
    common = {"pages": len(plan), "train_pages": n_train_pages, "val_pages": len(plan) - n_train_pages,
              "unique_cleans": len(cleans), "variants_per_page": args.variants_per_page,
              "real_bank": len(real_bank), "synth_bank": len(synth_bank),
              "total_boxes": n_boxes, "real_boxes": n_real, "synth_boxes": n_synth,
              "synth_ratio": round(n_synth / max(1, n_boxes), 4),
              "total_tiles": len(tile_src_all),
              "real_tiles": n_real_tiles, "synth_tiles": n_synth_tiles,
              "synth_tile_ratio": round(n_synth_tiles / max(1, len(tile_src_all)), 4),
              "tiles_per_box_mean": round(float(np.mean(tpb)), 3) if tpb else 0,
              "tiles_per_box_max": int(max(tpb)) if tpb else 0,
              "skipped_tiles_oob": sum(r["skipped_tiles"] for r in results),
              "tiny_boxes_lt32px": sum(r["tiny"] for r in results),
              "huge_boxes_gt60pct": sum(r["huge"] for r in results),
              "skipped_overlaps": sum(r["skipped"] for r in results),
              "synth_fallback_to_real": sum(r["fallback_real"] for r in results),
              "synth_reused_with_replacement": sum(r["reuse_synth"] for r in results),
              "synth_without_replacement": bool(synth_exact),
              "scale_mean": round(float(np.mean(scales)), 3) if scales else 0,
              "scale_min": round(float(np.min(scales)), 3) if scales else 0,
              "scale_max": round(float(np.max(scales)), 3) if scales else 0,
              "min_wm_long_side": MIN_WM_LONG_SIDE,
              "crop": f"native-tile pad{args.pad} stride{args.stride}",
              "pad": args.pad, "stride": args.stride, "slbr_size": args.slbr_size,
              "workers": args.workers, "dummy_cleans": use_dummy, "seed": args.seed}
    det_stats = {**common, "kind": "det",
                 "layout": "images/{train,val} + labels/{train,val} + data.yaml"}
    clean_stats = {**common, "kind": "clean", "layout": "CLWD train/test",
                   "train_crops": tr_crops, "test_crops": te_crops,
                   "crops": tr_crops + te_crops}
    (out_det / "stats.json").write_text(json.dumps(det_stats, indent=2), encoding="utf-8")
    (out_clean / "stats.json").write_text(json.dumps(clean_stats, indent=2), encoding="utf-8")
    print(json.dumps(clean_stats, indent=2))


if __name__ == "__main__":
    main()

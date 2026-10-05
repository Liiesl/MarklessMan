# %% [code] {"jupyter":{"outputs_hidden":false}}
# MarklessMan — dataset generation on Kaggle. SELF-CONTAINED: paste every cell
# into a Kaggle notebook and Run All. No .py files needed, trains nothing.
#
# What it does: builds TWO independent datasets from one shared page plan
# (same seed/split, split-safe per clean):
#   det:   YOLO pages  (<OUT_DET>/images/{train,val} + labels/{train,val} + data.yaml)
#   clean: SLBR native-256 tiles in NATIVE CLWD layout
#          (<OUT_CLEAN>/train/... + <OUT_CLEAN>/test/...), tiled exactly like
#          Rust inference (padded box + stride tiling, no resize)
# from clean manhwa pages + real Shimu watermark PNGs + pre-generated synthetic
# watermark bank (synthetic_wm.py --bank), all attached as Kaggle datasets.
#
# CONFIG — edit these, then Run All.
CLEAN_SLUG = "shinyunaa/marklessman-cleans"      # e.g. "yourname/marklessman-cleans"
WM_SLUG = "shinyunaa/marklessman-watermarks"     # e.g. "yourname/marklessman-watermarks"
SYNTH_SLUG = ""  # pre-generated synth bank dataset; "" = find it inside the
                 # same attached input as cleans/watermarks (wm_*.png bank dir).
                 # Ignored when SYNTH_GEN is True.
SYNTH_GEN = True  # generate the exact number of signature-unique synthetic
                  # watermarks needed for 50-50 at runtime (imports synthetic_wm.py
                  # from the WM dataset); the pre-generated dir is then unused
SYNTH_SAVE = "/kaggle/working/synth_bank"  # runtime-generated synth PNGs land here
OUT_DET = "/kaggle/working/det"                # det (YOLO) dataset lands here
OUT_CLEAN = "/kaggle/working/clean"            # clean (SLBR/CLWD) dataset lands here
SEED = 7
VARIANTS = 5                                   # N seeded variants per unique clean (split-safe)
WORKERS = 4                                    # Kaggle CPU notebook: 4 vCPU
VERIFY_PAGES = 100                             # quick check before the full run
VAL_RATIO = 0.2
SLBR_SIZE = 256
PAD = 64      # context padding around each tight box (mirrors Rust inference --pad)
STRIDE = 192  # tile stride (mirrors Rust inference --stride)
UPLOAD_DET_HANDLE = "shinyunaa/marklessman-det"
UPLOAD_CLEAN_HANDLE = "shinyunaa/marklessman-clean"
DO_UPLOAD = False                              # True -> kagglehub.dataset_upload both outputs

import subprocess
import sys
from pathlib import Path


def ensure_pkg(name: str) -> None:
    try:
        __import__(name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", name], check=True)


for _pkg in ("tqdm",):
    ensure_pkg(_pkg)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Fetch inputs via kagglehub (authenticated by default in Kaggle notebooks).
# If a slug is still a placeholder, fall back to attached /kaggle/input data.
# The synth bank lives on the SAME input dataset as the cleans/Shimu WMs:
# any dir with wm_*.png files (excluding preview/) outside the real-WM tree.
import kagglehub


def resolve_input(slug: str, fallback_substr: str) -> Path:
    if not slug.startswith("<"):
        return Path(kagglehub.dataset_download(slug))  # read-only cache is fine
    cands = [p for p in Path("/kaggle/input").rglob("*") if fallback_substr in p.parts]
    if not cands:
        raise SystemExit(f"Set the slug or attach an input dataset containing '{fallback_substr}'.")
    dirs = sorted({p if p.is_dir() else p.parent for p in cands}, key=lambda d: len(d.parts))
    return dirs[-1]


def resolve_synth(slug: str, wm_dir: Path, clean_dir: Path):
    """Locate the pre-generated synthetic bank (wm_*.png). None = absent."""
    if slug and not slug.startswith("<"):
        return Path(kagglehub.dataset_download(slug))
    # search attached inputs for synth-style files (same dataset as cleans/WMs)
    hits = [p for p in Path("/kaggle/input").rglob("wm_*.png")
            if "preview" not in p.parts and "Presets" not in p.parts]
    # prefer hits outside the real-WM tree so Shimu PNGs never count as synth
    outside = [p for p in hits if wm_dir not in p.parents]
    pool = outside or hits
    if pool:
        dirs = sorted({p.parent for p in pool}, key=lambda d: len(d.parts))
        return dirs[-1]
    # last resort: next to the resolved inputs (local-style layout)
    for base in (wm_dir.parent, clean_dir.parent):
        near = sorted(base.rglob("wm_*.png")) if base.exists() else []
        near = [p for p in near if "preview" not in p.parts]
        if near:
            return near[0].parent
    return None


clean_dir = resolve_input(CLEAN_SLUG, "cleans")
wm_dir = resolve_input(WM_SLUG, "Shimu WMs")
synth_dir = None if SYNTH_GEN else resolve_synth(SYNTH_SLUG, wm_dir, clean_dir)

# synthetic_wm.py ships INSIDE the WM dataset (user-added .py file); it is the
# runtime generator for unique synthetic watermarks when SYNTH_GEN is True.
synth_mod = None
if SYNTH_GEN:
    cands = sorted(wm_dir.rglob("synthetic_wm.py"))
    assert cands, f"synthetic_wm.py not found under {wm_dir} — add it to the WM dataset"
    sys.path.insert(0, cands[0].parent.as_posix())
    import synthetic_wm as synth_mod
    print(f"synthetic_wm loaded from {cands[0]}")

exts = {".jpg", ".jpeg", ".png", ".webp"}
n_cleans = sum(1 for p in Path(clean_dir).rglob("*") if p.suffix.lower() in exts)
n_wm = sum(1 for p in Path(wm_dir).rglob("*.png") if "Presets" not in p.parts)
n_synth = (sum(1 for p in Path(synth_dir).rglob("*.png") if "preview" not in p.parts)
           if synth_dir is not None else 0)
print(f"cleans: {clean_dir} ({n_cleans} images)")
print(f"watermarks (real): {wm_dir} ({n_wm} PNGs)")
if SYNTH_GEN:
    print(f"watermarks (synth): runtime-generated via synthetic_wm -> {SYNTH_SAVE}")
else:
    print(f"watermarks (synth): {synth_dir} ({n_synth} PNGs)")
assert n_cleans > 0 and n_wm > 0, "empty inputs, check slugs/attachments"
if not SYNTH_GEN and n_synth == 0:
    print("[warn] no synthetic bank found -> real-only fallback "
          "(50-50 disabled until the pre-generated bank is attached)")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Generator code (same logic as local generate_datasets.py).
import concurrent.futures
import io
import json
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

YOLO_CLASS = 0
MIN_WM_LONG_SIDE = 64
OVERLAP_ATTEMPTS = 25
CLWD_SUBS = ("Watermarked_image", "Watermark_free_image", "Mask", "Alpha", "Watermark")

ANCHORS = ["top-left", "top-center", "top-right", "center",
           "bottom-left", "bottom-center", "bottom-right", "random"]


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


def load_real_bank(wm_root):
    pngs = sorted([p for p in Path(wm_root).rglob("*.png") if "Presets" not in p.parts])
    if not pngs:
        raise SystemExit(f"No real watermark PNGs under {wm_root}")
    return pngs


def load_synth_bank(synth_root):
    if synth_root is None or not Path(synth_root).exists():
        return []
    return sorted([p for p in Path(synth_root).rglob("*.png")
                   if "preview" not in p.parts and "Presets" not in p.parts])


def collect_cleans(clean_dir):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in Path(clean_dir).rglob("*") if p.suffix.lower() in exts])


def plan_sources(n_pages, seed):
    """Per-page watermark-source slots with exact 50-50 global real:synth.

    Counts (1-3/page) come from an independent `seed:count:i` stream;
    labels are shuffled with `seed:sources` and dealt out in plan order.
    Workers loop over their slot list (no per-page randint), so the ratio
    holds regardless of worker count/scheduling. Skipped overlaps consume
    their slot (final placed ratio exact up to skips, reported in stats).
    """
    counts = [random.Random(f"{seed}:count:{i}").randint(1, 3) for i in range(n_pages)]
    total = sum(counts)
    n_synth = total // 2
    labels = ["real"] * (total - n_synth) + ["synth"] * n_synth
    r = random.Random(f"{seed}:sources")
    r.shuffle(labels)
    out = []
    it = iter(labels)
    for c in counts:
        out.append([next(it) for _ in range(c)])
    return out


def gen_unique_synth_bank(n, seed, out_dir):
    """Generate n signature-unique synthetic watermarks (deterministic).

    Uses the synthetic_wm module imported from the WM dataset (see SYNTH_GEN).
    ~0.01s each (12k ~= 3 min, single process). Files land in out_dir as
    wm_<i>_<style>.png so workers read them back like any pre-generated bank.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("wm_*.png")):
        try:
            stale.unlink()
        except OSError:
            pass
    rng = random.Random(f"{seed}:synthgen")
    # int-list seed: np<2 rejects str entropy (SeedSequence TypeError on Kaggle)
    np_rng = np.random.default_rng([seed, 0x51EED])
    seen = set()
    made = attempts = 0
    max_attempts = n * 25 + 100
    while made < n and attempts < max_attempts:
        attempts += 1
        wm, style = synth_mod.generate_synthetic_wm(rng, np_rng)
        sig = synth_mod.wm_signature(wm, style)
        if sig in seen:
            continue
        seen.add(sig)
        wm.save(out / f"wm_{made:05d}_{style}.png")
        made += 1
        if made % 1000 == 0:
            print(f"  synth bank: {made}/{n}")
    if made < n:
        print(f"[warn] synth bank shortfall: {made}/{n} unique "
              f"({attempts - made} collisions — raise retry budget)")
    print(f"[info] synth bank: {made}/{n} unique in {attempts} attempts -> {out}")
    return sorted(out.glob("wm_*.png"))


def deal_synth_picks(sources_per_page, synth_len, seed):
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


def anchor_position(anchor, pg_w, pg_h, wm_w, wm_h, rng, margin=8):
    if anchor == "random":
        return rng.randint(0, max(0, pg_w - wm_w)), rng.randint(0, max(0, pg_h - wm_h))
    xs = {"top-left": margin, "center": (pg_w - wm_w) // 2, "top-center": (pg_w - wm_w) // 2,
          "bottom-center": (pg_w - wm_w) // 2, "top-right": pg_w - wm_w - margin,
          "bottom-left": margin, "bottom-right": pg_w - wm_w - margin}
    ys = {"top-left": margin, "top-center": margin, "top-right": margin,
          "center": (pg_h - wm_h) // 2, "bottom-left": pg_h - wm_h - margin,
          "bottom-center": pg_h - wm_h - margin, "bottom-right": pg_h - wm_h - margin}
    x = int(np.clip(xs.get(anchor, 0) + rng.randint(-10, 10), 0, max(0, pg_w - wm_w)))
    y = int(np.clip(ys.get(anchor, 0) + rng.randint(-10, 10), 0, max(0, pg_h - wm_h)))
    return x, y


def boxes_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def overlay_rgba(page, wm, x, y):
    """Native PNG alpha composite. Returns (composited, alpha_full, wrgb_full)."""
    pg = np.asarray(page).astype(np.float32)
    wm_arr = np.asarray(wm).astype(np.float32)
    if wm_arr.shape[2] == 3:
        wm_arr = np.dstack([wm_arr, np.full(wm_arr.shape[:2], 255.0, dtype=np.float32)])
    H, W = pg.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + wm_arr.shape[1]), min(H, y + wm_arr.shape[0])
    if x1 <= x0 or y1 <= y0:
        return page, np.zeros((H, W), np.float32), np.zeros((H, W, 3), np.float32)
    patch = wm_arr[y0 - y:y0 - y + (y1 - y0), x0 - x:x0 - x + (x1 - x0)]
    a = patch[..., 3:4] / 255.0
    rgb = patch[..., :3]
    pg[y0:y1, x0:x1] = a * rgb + (1.0 - a) * pg[y0:y1, x0:x1]
    alpha_full = np.zeros((H, W), np.float32)
    alpha_full[y0:y1, x0:x1] = a[..., 0]
    wrgb_full = np.zeros((H, W, 3), np.float32)
    wrgb_full[y0:y1, x0:x1] = rgb
    return Image.fromarray(np.clip(pg, 0, 255).astype(np.uint8)), alpha_full, wrgb_full


def jpeg_bytes(im, q):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def process_page(page_idx, clean_path, split, ctx):
    """One det page + its CLWD crops. All randomness from per-page RNGs (deterministic)."""
    rng = random.Random(f"{ctx['seed']}:{page_idx}")
    real_bank, synth_bank = ctx["real_bank"], ctx["synth_bank"]
    out_det, out_clean = ctx["out_det"], ctx["out_clean"]
    S = ctx["slbr_size"]
    sources = ctx["picks_per_page"][page_idx]  # list of (src, bank_idx|None)
    cp = clean_path if clean_path is not None else rng.choice(ctx["cleans"])
    clean = Image.open(cp).convert("RGB")
    if clean.width > 900:
        clean = clean.resize((900, int(clean.height * 900 / clean.width)), Image.LANCZOS)
    pg_w, pg_h = clean.size
    page = clean.copy()
    alpha_accum = np.zeros((pg_h, pg_w), np.float32)
    wrgb_accum = np.zeros((pg_h, pg_w, 3), np.float32)
    boxes = []
    box_src = []
    scales = []
    tiny = huge = skipped = fallback_real = reuse_synth = 0

    # `sources` fixes the global 50-50; geometry below is IDENTICAL for real/synth.
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
        s = min(max(rng.gauss(1.0, 0.15), 0.7), 1.4)  # native-biased scale
        nw, nh = int(wm0.width * s), int(wm0.height * s)
        if nw > pg_w:
            k = pg_w / max(1, nw)
            nw, nh = max(8, int(nw * k)), max(8, int(nh * k))
        if nh > pg_h * 0.9:
            k = (pg_h * 0.9) / max(1, nh)
            nw, nh = max(8, int(nw * k)), max(8, int(nh * k))
        if max(nw, nh) < MIN_WM_LONG_SIDE:
            k = MIN_WM_LONG_SIDE / max(1, max(nw, nh))
            nw, nh = int(nw * k), int(nh * k)
        scales.append(nw / max(1, wm0.width))
        wm = wm0.resize((nw, nh), Image.LANCZOS)
        if rng.random() < 0.3:
            wm = wm.rotate(rng.uniform(-8, 8), expand=True, resample=Image.BICUBIC)
        wm_alpha = np.asarray(wm)[..., 3]
        ay, ax = np.where(wm_alpha > 5)
        if len(ax) == 0:
            continue
        tx0, ty0 = int(ax.min()), int(ay.min())
        tw, th = int(ax.max()) + 1 - tx0, int(ay.max()) + 1 - ty0
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
        wrgb_accum = np.where(a_full[..., None] > 0.02, w_full, wrgb_accum)
        tiny += int(min(bw, bh) < 32)
        huge += int(max(bw / pg_w, bh / pg_h) > 0.6)

    page = jpeg_bytes(page, rng.randint(75, 95))

    stem = f"page_{page_idx:05d}"
    page.save(out_det / "images" / split / f"{stem}.jpg", quality=92)
    with open(out_det / "labels" / split / f"{stem}.txt", "w") as f:
        for (bx, by, bw, bh) in boxes:
            xc = min(max((bx + bw / 2) / pg_w, 0.0), 1.0)
            yc = min(max((by + bh / 2) / pg_h, 0.0), 1.0)
            f.write(f"{YOLO_CLASS} {xc:.6f} {yc:.6f} "
                    f"{min(bw / pg_w, 1.0):.6f} {min(bh / pg_h, 1.0):.6f}\n")

    clwd_split = "train" if split == "train" else "test"
    pad, stride = ctx["pad"], ctx["stride"]
    page_np = np.asarray(page)
    clean_np = np.asarray(clean)
    alpha_u8 = (np.clip(alpha_accum, 0, 1) * 255).astype(np.uint8)
    mask_u8 = (alpha_accum > 0.1).astype(np.uint8) * 255
    wrgb_u8 = np.clip(wrgb_accum, 0, 255).astype(np.uint8)
    cids = []
    cid_src = []
    tiles_per_box = []
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
            cid = f"{page_idx:05d}_{k}_t{t}"
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


def run_gen(out_det_dir, out_clean_dir, clean_dir_p, wm_root_p, synth_root_p,
            seed, variants, workers, num_pages, val_ratio, slbr_size,
            pad=PAD, stride=STRIDE, gen_synth=SYNTH_GEN, synth_save=SYNTH_SAVE):
    out_det, out_clean = Path(out_det_dir), Path(out_clean_dir)
    rng = random.Random(seed)
    real_bank = load_real_bank(wm_root_p)
    dir_bank = load_synth_bank(synth_root_p)
    gen_here = bool(gen_synth and synth_mod is not None)
    if not gen_here and not dir_bank:
        print("[warn] synth bank empty -> real-only fallback for this run")
    cleans = collect_cleans(clean_dir_p)
    for split in ("train", "val"):
        (out_det / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_det / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for sub in CLWD_SUBS:
            (out_clean / split / sub).mkdir(parents=True, exist_ok=True)
    (out_det / "preview").mkdir(parents=True, exist_ok=True)
    (out_clean / "preview").mkdir(parents=True, exist_ok=True)

    if variants > 1 and cleans:
        order_c = cleans.copy()
        rng.shuffle(order_c)
        n_val_c = int(round(len(order_c) * val_ratio))
        val_set = set(order_c[:n_val_c])
        plan = []
        for cp in order_c:
            split = "val" if cp in val_set else "train"
            for _ in range(variants):
                plan.append((cp, split))
        rng.shuffle(plan)
        print(f"[info] {len(order_c)} unique cleans x {variants} variants = "
              f"{len(plan)} pages (split-safe per clean)")
    else:
        n_v = int(round(num_pages * val_ratio))
        plan = [(None, ("val" if i >= num_pages - n_v else "train")) for i in range(num_pages)]

    sources_per_page = plan_sources(len(plan), seed)

    # --- synthetic bank: runtime-generate exactly the slots needed (unique),
    # or reuse the pre-generated dir. Without-replacement dealing follows.
    synth_source = "dir"
    if gen_here:
        need = sum(1 for pg in sources_per_page for s in pg if s == "synth")
        print(f"[info] generating {need} unique synthetic watermarks -> {synth_save}")
        synth_bank = gen_unique_synth_bank(need, seed, synth_save)
        synth_source = "runtime"
    else:
        synth_bank = dir_bank
    picks_per_page, synth_exact = deal_synth_picks(sources_per_page, len(synth_bank), seed)

    ctx = {"real_bank": real_bank, "synth_bank": synth_bank, "cleans": cleans,
           "out_det": out_det, "out_clean": out_clean,
           "slbr_size": slbr_size, "pad": pad, "stride": stride,
           "use_dummy": False, "seed": seed,
           "picks_per_page": picks_per_page}
    print(f"[info] {len(plan)} pages on {workers} workers "
          f"(real={len(real_bank)} synth={len(synth_bank)}[{synth_source}] "
          f"unique={synth_exact})")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_page, i, cp, sp, ctx): i for i, (cp, sp) in enumerate(plan)}
        for fut in tqdm(concurrent.futures.as_completed(futs),
                         total=len(plan), desc="pages", unit="page"):
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
    n_train = sum(1 for _, s in plan if s == "train")
    n_boxes = sum(r["n_boxes"] for r in results)
    tr_crops = sum(len(r["cids"]) for r in results if r["split"] == "train")
    te_crops = sum(len(r["cids"]) for r in results if r["split"] == "val")
    common = {"pages": len(plan), "train_pages": n_train, "val_pages": len(plan) - n_train,
              "unique_cleans": len(cleans), "variants_per_page": variants,
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
              "synth_bank_source": synth_source,
              "scale_mean": round(float(np.mean(scales)), 3) if scales else 0,
              "scale_min": round(float(np.min(scales)), 3) if scales else 0,
              "scale_max": round(float(np.max(scales)), 3) if scales else 0,
              "min_wm_long_side": MIN_WM_LONG_SIDE,
              "crop": f"native-tile pad{pad} stride{stride}",
              "pad": pad, "stride": stride, "slbr_size": slbr_size,
              "workers": workers, "seed": seed}
    det_stats = {**common, "kind": "det",
                 "layout": "images/{train,val} + labels/{train,val} + data.yaml"}
    clean_stats = {**common, "kind": "clean", "layout": "CLWD train/test",
                   "train_crops": tr_crops, "test_crops": te_crops,
                   "crops": tr_crops + te_crops}
    (out_det / "stats.json").write_text(json.dumps(det_stats, indent=2), encoding="utf-8")
    (out_clean / "stats.json").write_text(json.dumps(clean_stats, indent=2), encoding="utf-8")
    print(json.dumps(clean_stats, indent=2))
    return clean_stats

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Verify run: 100 pages. Same seed => identical stats to a local run,
# which proves this notebook reproduces the exact same dataset.
run_gen(str(Path(OUT_DET).parent / "_verify100_det"),
        str(Path(OUT_CLEAN).parent / "_verify100_clean"),
        clean_dir, wm_dir, synth_dir,
        SEED, 1, WORKERS, VERIFY_PAGES, VAL_RATIO, SLBR_SIZE, PAD, STRIDE)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Full comfortable gen. ~25-35 min on 4 vCPU. Unattended. Two outputs.
run_gen(OUT_DET, OUT_CLEAN, clean_dir, wm_dir, synth_dir, SEED, VARIANTS, WORKERS,
        0, VAL_RATIO, SLBR_SIZE, PAD, STRIDE)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Persist det + clean as versioned Kaggle datasets for the training notebooks.
# Option A (explicit): kagglehub upload. Option B: "Save & Run All"
# (everything in /kaggle/working is kept as the notebook's output).
det_stats = json.loads((Path(OUT_DET) / "stats.json").read_text())
clean_stats = json.loads((Path(OUT_CLEAN) / "stats.json").read_text())
print(json.dumps(det_stats, indent=2))
print(json.dumps(clean_stats, indent=2))

if DO_UPLOAD:
    up_det = kagglehub.dataset_upload(
        UPLOAD_DET_HANDLE, OUT_DET,
        version_notes=f"det seed={SEED} variants={VARIANTS} pages={det_stats['pages']}")
    print("uploaded det:", up_det)
    up_clean = kagglehub.dataset_upload(
        UPLOAD_CLEAN_HANDLE, OUT_CLEAN,
        version_notes=f"clean seed={SEED} variants={VARIANTS} "
                      f"pages={clean_stats['pages']} crops={clean_stats['crops']} "
                      f"synth_ratio={clean_stats['synth_ratio']}")
    print("uploaded clean:", up_clean)
else:
    print(f"DO_UPLOAD=False -> use Save & Run All to keep {OUT_DET} + {OUT_CLEAN} as outputs.")

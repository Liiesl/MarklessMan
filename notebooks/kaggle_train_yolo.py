# %% [code] {"jupyter":{"outputs_hidden":false}}
# MarklessMan — Stage-1 YOLO26s training on Kaggle. SELF-CONTAINED: paste every
# cell into a Kaggle notebook and Run All. Trains nothing locally.
#
# Input:  shinyunaa/marklessman-det  (det/{images,labels}/{train,val} + data.yaml)
# Output: /kaggle/working/runs/detect/markless_yolo26s/{best.pt,last.pt,...}
#         + /kaggle/working/yolo_val.json (val metrics snapshot)
#
# CONFIG — edit these, then Run All.
DATASET_SLUG = "shinyunaa/marklessman-det-dataset"   # det dataset (images/ + labels/ + data.yaml + stats.json)
BASE_MODEL = "yolo26s.pt"                   # COCO-pretrained small; auto-downloaded by ultralytics
EPOCHS = 100
IMGSZ = 640
BATCH_PER_GPU = 32                          # per-GPU batch (s@640 fits T4 16GB @9.3GiB); total = x N_GPU
WORKERS = 8                                 # 2xT4 Kaggle: 8 total; single-GPU: 4 is enough
PATIENCE = 30                               # early-stop patience
SEED = 7
PROJECT = "/kaggle/working/runs"            # ultralytics project dir
RUN_NAME = "markless_yolo26s"
CONF_VAL = 0.25                             # val conf threshold for P/R snapshot
UPLOAD_HANDLE = "shinyunaa/marklessman-yolo26s"
DO_UPLOAD = False                           # True -> kagglehub.dataset_upload at the end
OOM_TEST = True                             # True -> 2-epoch smoke, same batch/imgsz; OOMs in minutes

import subprocess
import sys
from pathlib import Path


def ensure_pkg(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name],
                       check=True)


for _imp, _pip in (("ultralytics", "ultralytics"), ("yaml", "pyyaml")):
    ensure_pkg(_imp, _pip)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Fetch v1: prefer the attached input dataset under /kaggle/input (no download).
# Fallback to kagglehub download only if nothing is attached.
import shutil

import kagglehub
import yaml


def resolve_v1_yolo() -> Path:
    def _is_det_root(d: Path) -> bool:
        return (d / "data.yaml").exists() and (d / "images" / "train").exists()
    hits = [p for p in Path("/kaggle/input").rglob("data.yaml")
            if _is_det_root(p.parent)]
    if hits:
        print(f"using attached input: {hits[0]}")
        return hits[0].parent
    print("nothing under /kaggle/input — downloading via kagglehub")
    _ds = Path(kagglehub.dataset_download(DATASET_SLUG))
    cands = [p for p in _ds.rglob("data.yaml") if _is_det_root(p.parent)]
    assert cands, "no det data.yaml found — check DATASET_SLUG / attachment"
    return cands[0].parent


_v1yolo = resolve_v1_yolo()

work_yolo = Path("/kaggle/working/yolo")
if work_yolo.exists():
    shutil.rmtree(work_yolo)
shutil.copytree(_v1yolo, work_yolo)
print(f"copied -> {work_yolo}")

# Rewrite data.yaml: generated data.yaml carries a local absolute path
# (e.g. E:/...) which is invalid on Kaggle. Point it at /kaggle/working/yolo.
dy = work_yolo / "data.yaml"
cfg = yaml.safe_load(dy.read_text())
cfg["path"] = work_yolo.resolve().as_posix()
cfg["train"] = "images/train"
cfg["val"] = "images/val"
cfg["names"] = {0: "watermark"}
dy.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dy.read_text())

n_tr = len(sorted((work_yolo / "images" / "train").glob("*.jpg")))
n_va = len(sorted((work_yolo / "images" / "val").glob("*.jpg")))
n_ltr = len(sorted((work_yolo / "labels" / "train").glob("*.txt")))
n_lva = len(sorted((work_yolo / "labels" / "val").glob("*.txt")))
print(f"train images/labels: {n_tr}/{n_ltr}  val images/labels: {n_va}/{n_lva}")
assert n_tr > 0 and n_va > 0 and n_tr == n_ltr and n_va == n_lva, "image/label mismatch"

# spot-check: every label line is single-class YOLO format
bad = 0
for t in sorted((work_yolo / "labels" / "train").glob("*.txt"))[:200]:
    for ln in t.read_text().splitlines():
        if not ln.strip():
            continue
        p = ln.split()
        if len(p) != 5 or p[0] != "0":
            bad += 1
            break
print(f"label spot-check: {bad} bad files in first 200")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# GPU check — Kaggle T4x2 gives 2x Tesla T4 (16GB each). Use ALL of them via
# DDP (device=[0,1]); effective batch = BATCH_PER_GPU x N_GPU. Fail fast on CPU.
import torch

N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpus={N_GPU}")
for _i in range(N_GPU):
    print(f"  gpu{_i}: {torch.cuda.get_device_name(_i)}  "
          f"mem={torch.cuda.get_device_properties(_i).total_memory / 1e9:.1f}GB")
if N_GPU == 0:
    print("  <-- ENABLE GPU: Settings > Accelerator > GPU T4 x2")
assert N_GPU > 0, "no GPU — enable Kaggle GPU accelerator and re-run"

DEVICE = 0 if N_GPU == 1 else list(range(N_GPU))  # DDP when 2xT4
BATCH = BATCH_PER_GPU * N_GPU
print(f"using DEVICE={DEVICE}  total batch={BATCH} ({BATCH_PER_GPU}/GPU)")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Train YOLO26s. Single class, tall manhwa pages: rect=False + mosaic with
# close_mosaic=10 (ultralytics default recipe for small models). amp=True uses
# T4 Tensor Cores. 2xT4 DDP ~= 1.7x single-GPU speed.
# Time: ~1-2h on 2xT4 for ~10k pages x 100 epochs (early-stop often kicks in).
# If DDP fails in-notebook, fallback: set DEVICE = 0 manually and re-run.
from ultralytics import YOLO

EPOCHS_RUN = 2 if OOM_TEST else EPOCHS
RUN_NAME_RUN = f"{RUN_NAME}_oom" if OOM_TEST else RUN_NAME
print(f"OOM_TEST={OOM_TEST} -> epochs={EPOCHS_RUN} name={RUN_NAME_RUN} batch={BATCH}")
model = YOLO(BASE_MODEL)
results = model.train(
    data=(work_yolo / "data.yaml").as_posix(),
    epochs=EPOCHS_RUN,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    workers=WORKERS,
    single_cls=True,
    rect=False,
    close_mosaic=10,
    optimizer="auto",
    amp=True,
    seed=SEED,
    device=DEVICE,
    cache=False,
    project=PROJECT,
    name=RUN_NAME_RUN,
    exist_ok=True,
    verbose=True,
)
print(f"train done -> {Path(PROJECT) / RUN_NAME_RUN}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Validate best.pt on the held-out split + snapshot metrics to JSON.
# Synthetic watermarks should give high mAP; low numbers => data.yaml/labels issue.
import json

best = Path(PROJECT) / RUN_NAME_RUN / "weights" / "best.pt"
assert best.exists(), f"missing {best}"
val_model = YOLO(best.as_posix())
metrics = val_model.val(data=(work_yolo / "data.yaml").as_posix(), imgsz=IMGSZ,
                        conf=CONF_VAL, device=DEVICE, verbose=True)
snap = {
    "best": best.as_posix(),
    "map50": round(float(metrics.box.map50), 4),
    "map50_95": round(float(metrics.box.map), 4),
    "precision": round(float(metrics.box.mp), 4),
    "recall": round(float(metrics.box.mr), 4),
    "fitness": round(float(metrics.fitness), 4),
    "epochs": EPOCHS_RUN, "imgsz": IMGSZ, "seed": SEED, "dataset": DATASET_SLUG,
}
Path("/kaggle/working/yolo_val.json").write_text(json.dumps(snap, indent=2))
print(json.dumps(snap, indent=2))
print(f"artifacts: {(Path(PROJECT) / RUN_NAME_RUN).as_posix()}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Persist weights. Option A (explicit): kagglehub upload. Option B: "Save & Run
# All" (everything under /kaggle/working is kept as the notebook's output).
if DO_UPLOAD:
    up = kagglehub.dataset_upload(
        UPLOAD_HANDLE, str(Path(PROJECT) / RUN_NAME_RUN),
        version_notes=f"yolo26s ep={EPOCHS_RUN} imgsz={IMGSZ} "
                      f"mAP50={snap['map50']} mAP50-95={snap['map50_95']}")
    print("uploaded:", up)
else:
    print(f"DO_UPLOAD=False -> use Save & Run All to keep {Path(PROJECT) / RUN_NAME_RUN}")

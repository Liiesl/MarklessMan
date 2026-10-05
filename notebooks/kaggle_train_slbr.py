# %% [code] {"jupyter":{"outputs_hidden":false}}
# MarklessMan — Stage-2 SLBR finetune on Kaggle. SELF-CONTAINED: paste every
# cell into a Kaggle notebook and Run All. Trains nothing locally.
#
# Input:  shinyunaa/marklessman-clean  (native CLWD layout:
#         <root>/train/{Watermarked_image,Watermark_free_image,Mask,Alpha,Watermark}/
#         <root>/test/{...}/, native 256 tiles matching inference tiling,
#         no txt files, no reshape)
#         + CLWD-pretrained model_best.pth.tar (see CLWD_CKPT_SLUG / CLWD_CKPT_URL)
# Output: /kaggle/working/slbr_ckpt/markless_slbr/{model_best.pth.tar,checkpoint.pth.tar}
#
# Method: clone bcmi/SLBR-Visible-Watermark-Removal at runtime, point its
# train.py straight at the clean dataset (--dataset_dir <root>) since the
# generator already emits the CLWD layout CLWDDataset expects, then run with
# --resume <CLWD weights>. BasicModel.resume() loads only the weights and
# keeps a FRESH optimizer with our lower finetune LR, so --resume here acts
# as finetune (the repo's --finetune flag is a no-op).
#
# MOCK_RUN=True: 30-second dry run. Skips ALL downloads (synthetic 8+4 crops,
# no CLWD resume, no VGG weight fetch) and runs 1 epoch + eval + preview to
# prove the pipeline wiring before spending a GPU-quota slot. Flip to False
# for the real 30-epoch finetune.
#
# CONFIG — edit these, then Run All.
DATASET_SLUG = "shinyunaa/marklessman-clean-dataset"   # clean dataset (native CLWD train/test)
CLWD_CKPT_SLUG = "shinyunaa/slbr-pretrained"                         # e.g. "yourname/slbr-clwd-pretrain" (contains model_best.pth.tar); "" to skip
CLWD_CKPT_URL = ""                          # fallback direct URL (gdown-compatible Drive link or https .pth.tar); "" to skip
CLWD_CKPT_INPUT_HINT = "model_best"         # fallback: file containing this substring under /kaggle/input
REPO_URL = "https://github.com/bcmi/SLBR-Visible-Watermark-Removal.git"
REPO_DIR = "/kaggle/working/slbr_repo"
DATA_CLWD = ""  # resolved at runtime to the native CLWD clean root (no reshape/copy)
CKPT_DIR = "/kaggle/working/slbr_ckpt"
RUN_NAME = "markless_slbr"
# Finetune hyperparams (pretrain was lr=1e-3/batch=8/100ep; finetune = 10x lower LR).
# Batches are PER-GPU under DDP (torchrun, one process per T4; no DataParallel
# gather on GPU-0, so per-GPU footprint == peak footprint). 2xT4 with 4 train /
# 2 test per GPU => 8 / 4 effective (paper used batch 8 on one GPU).
# ~20k crops / 8 ~= 2500 iters/ep; 30ep ~= 6-9h on 2xT4. If OOM -> halve per-GPU.
EPOCHS = 7
TRAIN_BATCH_PER_GPU = 6
TEST_BATCH_PER_GPU = 4
LR = 1.2e-4
SCHEDULE = "3"                             # lr x0.1 at ep 3 (--gamma 0.1)
WORKERS = 4
BETA1 = 0.5                                 # paper value (options default 0.9 is for BasicModel)
BETA2 = 0.999
SEED = 7
UPLOAD_HANDLE = "shinyunaa/marklessman-slbr"
DO_UPLOAD = False                           # True -> kagglehub.dataset_upload at the end
MOCK_RUN = True                             # <-- SET TO True FOR A 30-SECOND DRY RUN
MOCK_N_TRAIN = 8                            # synthetic crops (no download, no ckpt fetch)
MOCK_N_VAL = 4
MOCK_SIZE = 64                              # px; upscaled to 256 by --preprocess resize

import subprocess
import sys
from pathlib import Path


def ensure_pkg(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name],
                       check=True)


# NOTE: do NOT `pip install -r requirements.txt` — it pins torch==1.6 /
# albumentations==0.4.5 / skimage 0.17 which break the Kaggle image.
# Kaggle already ships torch 2.x + torchvision; we add only the small extras.
for _imp, _pip in (("progress", "progress"), ("tensorboardX", "tensorboardX"),
                   ("albumentations", "albumentations"), ("skimage", "scikit-image"),
                   ("gdown", "gdown")):
    ensure_pkg(_imp, _pip)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Fetch v1: prefer the attached input dataset under /kaggle/input (no download).
# Fallback to kagglehub download only if nothing is attached.
import kagglehub


def resolve_clean_clwd() -> Path:
    """Find the native CLWD clean root (<root>/train/Watermarked_image).

    Works with attached /kaggle/input data or a kagglehub download. No
    train.txt/val.txt, no reshape: the generator already emits train/test.
    """
    hits = [p for p in Path("/kaggle/input").rglob("Watermarked_image") if p.is_dir()
            and p.parent.name in ("train", "test") and (p.parent.parent / "train").exists()]
    if hits:
        _root = hits[0].parent.parent
        print(f"using attached input: {_root}")
        return _root
    print("nothing under /kaggle/input — downloading via kagglehub")
    _ds = Path(kagglehub.dataset_download(DATASET_SLUG))
    _hit = next((p for p in _ds.rglob("Watermarked_image") if p.is_dir()
                 and p.parent.name in ("train", "test")), None)
    assert _hit is not None, "no train/Watermarked_image found — check DATASET_SLUG / attachment"
    _root = _hit.parent.parent if _hit.parent.name in ("train", "test") else _hit.parent
    return _root


def _make_mock_v1(n_train: int = MOCK_N_TRAIN, n_val: int = MOCK_N_VAL,
                    size: int = MOCK_SIZE) -> Path:
    """Synthetic native-CLWD tree — no network, no 6GB download.

    Layout + formats mirror the real clean dataset (JPG for images, PNG for
    mask/alpha/watermark) so CLWDDataset parses it identically.
    """
    import cv2 as _cv2
    import numpy as _np
    _root = Path("/kaggle/working/mock_clean")
    _subs = ("Watermarked_image", "Watermark_free_image", "Mask", "Alpha", "Watermark")
    for _split in ("train", "test"):
        for _s in _subs:
            (_root / _split / _s).mkdir(parents=True, exist_ok=True)
    _rng = _np.random.default_rng(7)

    def _one(_cid: str, _split: str) -> None:
        _free = _rng.integers(0, 256, (size, size, 3), dtype=_np.uint8)
        _wm = _np.full((size, size, 3), 255, dtype=_np.uint8)  # white overlay
        _m = _np.zeros((size, size), dtype=_np.uint8)
        _m[size // 4:3 * size // 4, size // 4:3 * size // 4] = 255
        _a = (_m.astype(_np.float32) / 255.0 * 0.6)[..., None]
        _marked = (_free.astype(_np.float32) * (1 - _a)
                   + _wm.astype(_np.float32) * _a).astype(_np.uint8)
        _cv2.imwrite(str(_root / _split / "Watermark_free_image" / f"{_cid}.jpg"), _free)
        _cv2.imwrite(str(_root / _split / "Watermarked_image" / f"{_cid}.jpg"), _marked)
        _cv2.imwrite(str(_root / _split / "Mask" / f"{_cid}.png"), _m)
        _cv2.imwrite(str(_root / _split / "Alpha" / f"{_cid}.png"), _m)
        _cv2.imwrite(str(_root / _split / "Watermark" / f"{_cid}.png"), _wm)

    _tr = [f"mock{i:04d}" for i in range(n_train)]
    _va = [f"mock9{i:03d}" for i in range(n_val)]
    for _cid in _tr:
        _one(_cid, "train")
    for _cid in _va:
        _one(_cid, "test")
    print(f"mock clean (CLWD): {n_train} train + {n_val} test crops @ {_root}")
    return _root

def find_ckpt() -> Path:
    if CLWD_CKPT_SLUG:
        d = Path(kagglehub.dataset_download(CLWD_CKPT_SLUG))
        hits = sorted(d.rglob("*.pth.tar")) + sorted(d.rglob("*.pth"))
        assert hits, f"no checkpoint found in {CLWD_CKPT_SLUG}"
        return hits[0]
    if CLWD_CKPT_URL:
        out = Path("/kaggle/working/clwd_pretrain.pth.tar")
        if not out.exists():
            if "drive.google" in CLWD_CKPT_URL:
                import gdown
                gdown.download(CLWD_CKPT_URL, str(out), quiet=False)
            else:
                subprocess.run(["wget", "-q", "-O", str(out), CLWD_CKPT_URL], check=True)
        return out
    hits = [p for p in Path("/kaggle/input").rglob("*.pth.tar")
            if CLWD_CKPT_INPUT_HINT in p.name] or \
           [p for p in Path("/kaggle/input").rglob("*.pth")
            if CLWD_CKPT_INPUT_HINT in p.name]
    assert hits, ("no CLWD checkpoint: set CLWD_CKPT_SLUG or CLWD_CKPT_URL, "
                  "or attach a file with 'model_best' in its name")
    return hits[0]

if MOCK_RUN:
    print("MOCK_RUN=True — skipping dataset + checkpoint downloads")
    v1_slbr = _make_mock_v1()
    clwd_ckpt = None
else:
    v1_slbr = resolve_clean_clwd()
    clwd_ckpt = find_ckpt()
print(f"clean CLWD root: {v1_slbr}")
# The generator emits native CLWD layout: use it directly, no reshape/symlinks.
DATA_CLWD = str(v1_slbr)
if clwd_ckpt is not None:
    # Inspect the resume checkpoint for the log (epoch matters: the patched
    # resume() no longer adopts it — finetune always starts at epoch 0 unless
    # RESUME_FROM continuation passes an explicit --start-epoch below).
    try:
        import torch as _torch_fetch
        _info = _torch_fetch.load(str(clwd_ckpt), map_location="cpu",
                                  weights_only=True)
        print(f"CLWD ckpt: epoch={_info.get('epoch')} "
              f"best={_info.get('best_acc')} nets={_info.get('nets')} "
              f"keys={len(_info.get('state_dict', {}))}")
    except Exception as _e:
        print(f"(could not inspect ckpt: {_e})")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Clone SLBR repo (runtime dependency, no .py files needed beforehand).
if not (Path(REPO_DIR) / "train.py").exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
else:
    print("repo cached")
print(sorted(p.name for p in Path(REPO_DIR).iterdir()))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Validate the native CLWD layout the generator emits (no reshape/symlinks):
#   <root>/{train,test}/{Watermarked_image,Watermark_free_image,Mask,Alpha,Watermark}/
# NOTE: repo quirk — the "val" split is named `test/` (CLWDDataset maps
# 'val' -> <root>/test/). The clean dataset already uses that naming.
SUBS = ("Watermarked_image", "Watermark_free_image", "Mask", "Alpha", "Watermark")
EXT = {"Watermarked_image": ".jpg", "Watermark_free_image": ".jpg",
       "Mask": ".png", "Alpha": ".png", "Watermark": ".png"}


def validate_clwd(root: Path) -> dict:
    counts = {}
    for split in ("train", "test"):
        ids = sorted((root / split / "Watermarked_image").glob("*.jpg"))
        assert ids, f"empty {root}/{split}/Watermarked_image — check DATASET_SLUG / attachment"
        missing = 0
        for src in ids[:2000]:
            for sub in SUBS:
                if not (root / split / sub / f"{src.stem}{EXT[sub]}").exists():
                    missing += 1
                    break
        counts[split] = len(ids)
        print(f"{split}: {len(ids)} crops, missing companions in first 2000: {missing}")
        assert missing == 0, f"incomplete CLWD files under {root}/{split}"
    print(f"train crops: {counts['train']}  test(val) crops: {counts['test']}")
    return counts


counts = validate_clwd(Path(DATA_CLWD))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# GPU check. Kaggle T4x2 gives 2 GPUs; training uses DDP via torchrun
# (one process per GPU, DistributedSampler, whole-net DDP wrap in train.py).
# --train-batch / --test-batch are PER-GPU under DDP; effective = per-GPU x N_GPU.
import torch

N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpus={N_GPU}")
for _i in range(N_GPU):
    print(f"  gpu{_i}: {torch.cuda.get_device_name(_i)}  "
          f"mem={torch.cuda.get_device_properties(_i).total_memory / 1e9:.1f}GB")
if N_GPU == 0:
    print("  <-- ENABLE GPU: Settings > Accelerator > GPU T4 x2")
assert N_GPU > 0, "no GPU — enable Kaggle GPU accelerator and re-run"

TRAIN_BATCH = TRAIN_BATCH_PER_GPU
TEST_BATCH = TEST_BATCH_PER_GPU
if MOCK_RUN:
    # 30-second dry run: 1 epoch on 12 tiny crops, no VGG download
    # (VGGLoss pulls vgg16 weights; lambdas zeroed in the train cell).
    EPOCHS, SCHEDULE = 10, "100"   # schedule step beyond the single epoch
    TRAIN_BATCH = TEST_BATCH = 1  # per-GPU; DDP total = N_GPU
    WORKERS = 0
    RUN_NAME = "mock_slbr"
    print(f"MOCK_RUN — dry run only: epochs=1 batch={TRAIN_BATCH}/GPU workers=0 name={RUN_NAME}")
print(f"ddp nproc={N_GPU}  train batch={TRAIN_BATCH}/GPU "
      f"(effective {TRAIN_BATCH * N_GPU})  test batch={TEST_BATCH}/GPU")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Compat shims for running the 2021-era repo on a 2026 Kaggle image.
# Must run BEFORE importing anything from the repo.
# ROOT CAUSE of the reported crash: the old in-memory-only
# `setattr(skimage.measure, ...)` patched just the parent notebook process,
# but `train.py`/`test.py` run via `subprocess.run([sys.executable, ...])`
# in FRESH interpreters that never see it -> ImportError on
# `from skimage.measure import compare_psnr,compare_ssim` (removed in
# skimage>=0.19, moved to skimage.metrics). Fix = patch ON DISK so every
# subprocess picks it up, plus a sitecustomize.py belt-and-suspenders.
import os as _os

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
_os.chdir(REPO_DIR)
# sitecustomize.py in the script dir does NOT auto-load (site.py runs before
# the script dir lands on sys.path), so export REPO_DIR on PYTHONPATH: every
# `train.py`/`test.py` subprocess then auto-imports it at startup.
_pp = _os.environ.get("PYTHONPATH", "")
if REPO_DIR not in _pp.split(_os.pathsep):
    _os.environ["PYTHONPATH"] = REPO_DIR + (_os.pathsep + _pp if _pp else "")
# Fragmentation guard for the 15GB T4 (suggested by the OOM error itself);
# inherited by every train.py/test.py subprocess.
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
print(f"cwd={_os.getcwd()}  PYTHONPATH={_os.environ['PYTHONPATH']}")

_SK_COMPAT_IMPORT = (
    "try:\n"
    "    from skimage.measure import compare_psnr, compare_ssim\n"
    "except ImportError:  # skimage>=0.19: moved to skimage.metrics\n"
    "    import numpy as _np\n"
    "    from skimage.metrics import (\n"
    "        peak_signal_noise_ratio as _psnr,\n"
    "        structural_similarity as _ssim,\n"
    "    )\n"
    "    def _infer_range(a, b, data_range):\n"
    "        if data_range is not None:\n"
    "            return data_range\n"
    "        for _x in (a, b):\n"
    "            _dt = getattr(_x, 'dtype', None)\n"
    "            if _dt is not None and _np.issubdtype(_dt, _np.integer):\n"
    "                _info = _np.iinfo(_dt)\n"
    "                return float(_info.max - _info.min)\n"
    "        return 255.0 if (float(_np.asarray(a).max(initial=0)) > 1.5 or float(_np.asarray(b).max(initial=0)) > 1.5) else 1.0\n"
    "    def compare_psnr(im_true, im_test, data_range=None, **kw):\n"
    "        return _psnr(im_true, im_test, data_range=_infer_range(im_true, im_test, data_range), **kw)\n"
    "    def compare_ssim(X, Y, data_range=None, multichannel=False, channel_axis=None, win_size=None, **kw):\n"
    "        if channel_axis is None and multichannel:\n"
    "            channel_axis = -1  # old multichannel=True meant last axis\n"
    "        _kw = dict(kw)\n"
    "        if win_size is not None:\n"
    "            _kw.setdefault('win_size', win_size)\n"
    "        return _ssim(X, Y, data_range=_infer_range(X, Y, data_range), channel_axis=channel_axis, **_kw)\n"
)

_SK_TEST_IMPORT = (
    "try:\n"
    "    from skimage.measure import compare_ssim as ssim\n"
    "except ImportError:  # skimage>=0.19\n"
    "    import numpy as _np\n"
    "    from skimage.metrics import structural_similarity as _ssim\n"
    "    def ssim(X, Y, data_range=None, multichannel=False, channel_axis=None, win_size=None, **kw):\n"
    "        if channel_axis is None and multichannel:\n"
    "            channel_axis = -1\n"
    "        if data_range is None:\n"
    "            data_range = 255.0 if (float(_np.asarray(X).max(initial=0)) > 1.5 or float(_np.asarray(Y).max(initial=0)) > 1.5) else 1.0\n"
    "        _kw = dict(kw)\n"
    "        if win_size is not None:\n"
    "            _kw.setdefault('win_size', win_size)\n"
    "        return _ssim(X, Y, data_range=data_range, channel_axis=channel_axis, **_kw)\n"
)

_ALBU_COMPAT_IMPORT = (
    "try:\n"
    "    from albumentations import HorizontalFlip, RandomResizedCrop, Compose, DualTransform\n"
    "    import albumentations.augmentations.transforms as transforms\n"
    "except ImportError:  # albumentations>=2.x restructured submodules\n"
    "    import albumentations as _A\n"
    "    HorizontalFlip = _A.HorizontalFlip\n"
    "    RandomResizedCrop = _A.RandomResizedCrop\n"
    "    Compose = _A.Compose\n"
    "    try:\n"
    "        DualTransform = _A.DualTransform\n"
    "    except AttributeError:\n"
    "        from albumentations.core.transforms_interface import DualTransform\n"
    "    import albumentations as transforms  # Resize, ToGray live at top level\n"
)

_HCOMPOSE_COMPAT_OLD = (
    "        if additional_targets is None:\n"
    "            additional_targets = {\n"
    "                'real': 'image',\n"
    "                # 'mask': 'mask'\n"
    "            }\n"
    "        self.additional_targets = additional_targets\n"
    "        super().__init__(transforms, *args, additional_targets=additional_targets, **kwargs)"
)
_HCOMPOSE_COMPAT_NEW = (
    "        if additional_targets is None:\n"
    "            additional_targets = {\n"
    "                'real': 'image',\n"
    "                # 'mask': 'mask'\n"
    "            }\n"
    "        # NOTE: do NOT assign self.additional_targets — it is a read-only\n"
    "        # property on albumentations>=2.x Compose (base class exposes it).\n"
    "        super().__init__(transforms, *args, additional_targets=additional_targets, **kwargs)"
)

_RESUME_REMAP = (
    "        # The checkpoint may come from a differently-wrapped model: a\n"
    "        # single-GPU save has `encoder.conv...` while a DataParallel save has\n"
    "        # `encoder.module.conv...`. Remap the `.module.` infix in either\n"
    "        # direction, then load strict so genuine architecture mismatches\n"
    "        # still fail loudly.\n"
    "        _own_keys = set(self.model.state_dict().keys())\n"
    "        _ckpt_sd = current_checkpoint['state_dict']\n"
    "        if set(_ckpt_sd.keys()) != _own_keys:\n"
    "            from collections import OrderedDict as _OD\n"
    "            _remapped = _OD()\n"
    "            _n_map = 0\n"
    "            for _k, _v in _ckpt_sd.items():\n"
    "                if _k in _own_keys:\n"
    "                    _remapped[_k] = _v\n"
    "                    continue\n"
    "                _parts = _k.split('.')\n"
    "                _cand = '.'.join([_parts[0], 'module'] + _parts[1:])\n"
    "                if _cand in _own_keys:\n"
    "                    _remapped[_cand] = _v\n"
    "                    _n_map += 1\n"
    "                    continue\n"
    "                _cand0 = 'module.' + _k  # whole-net DataParallel save\n"
    "                if _cand0 in _own_keys:\n"
    "                    _remapped[_cand0] = _v\n"
    "                    _n_map += 1\n"
    "                    continue\n"
    "                if 'module' in _parts:\n"
    "                    _cand2 = '.'.join(_p for _p in _parts if _p != 'module')\n"
    "                    if _cand2 in _own_keys:\n"
    "                        _remapped[_cand2] = _v\n"
    "                        _n_map += 1\n"
    "                        continue\n"
    "                _remapped[_k] = _v\n"
    "            _ckpt_sd = _remapped\n"
    "            print(\"=> remapped {} DataParallel `.module.` keys\".format(_n_map))\n"
    "        self.model.load_state_dict(_ckpt_sd, strict=True)\n"
)

_RESUME_EPOCH_OLD = (
    "        if self.args.start_epoch == 0:\n"
    "            self.args.start_epoch = current_checkpoint['epoch']\n"
)
_RESUME_EPOCH_NEW = (
    "        # NOTE: explicit --start-epoch is the ONLY way to set the start epoch\n"
    "        # (auto-adopt removed: finetuning from foreign weights must start at 0).\n"
)

_VGG_COMPAT = (
    "        try:  # torchvision>=0.13 weights API (pretrained= removed in 0.15)\n"
    "            vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)\n"
    "        except (AttributeError, TypeError):\n"
    "            vgg16 = models.vgg16(pretrained=True)\n"
)

# --- DDP conversion: drop per-submodule DataParallel (GPU-0 gather OOM on
# T4x2), use whole-net DistributedDataParallel via torchrun. Applied on disk
# to the fresh Kaggle clone, same mechanism as the compat shims above. ---
_DDP_MULTIGPU_OLD = (
    "    def multi_gpu(self):\n"
    "        self.encoder = nn.DataParallel(self.encoder, device_ids=range(torch.cuda.device_count()))\n"
    "        self.shared_decoder = nn.DataParallel(self.shared_decoder, device_ids=range(torch.cuda.device_count()))\n"
    "        self.coarse_decoder = nn.DataParallel(self.coarse_decoder, device_ids=range(torch.cuda.device_count()))\n"
    "        if self.refinement is not None:\n"
    "            self.refinement = nn.DataParallel(self.refinement, device_ids=range(torch.cuda.device_count()))\n"
    "        return\n"
)
_DDP_MULTIGPU_NEW = (
    "    def multi_gpu(self):\n"
    "        # DDP: no-op. Whole-net DDP wrap happens in train.py main() after init.\n"
    "        # Kept so single-GPU inference paths calling multi_gpu() keep working.\n"
    "        return\n"
)

_DDP_DEVICE_NEW = (
    "        # DDP: per-rank device (torchrun sets LOCAL_RANK). Single-GPU unchanged.\n"
    "        _lr = int(os.environ.get('LOCAL_RANK', '0'))\n"
    "        if os.environ.get('SLBR_CPU', '0') == '1':\n"
    "            self.device = torch.device('cpu')\n"
    "        elif torch.cuda.is_available():\n"
    "            torch.cuda.set_device(_lr % max(1, torch.cuda.device_count()))\n"
    "            self.device = torch.device('cuda', torch.cuda.current_device())\n"
    "        else:\n"
    "            self.device = torch.device('cpu')\n"
)

_DDP_NODP_OLD = (
    "        if self.count_gpu > 1 : # multiple\n"
    "            # self.model = DataParallelModel(self.model, device_ids=range(torch.cuda.device_count()))\n"
    "            # self.loss = DataParallelCriterion(self.loss, device_ids=range(torch.cuda.device_count()))\n"
    "            self.model.multi_gpu()\n"
)
_DDP_NODP_NEW = (
    "        # DDP: per-submodule DataParallel removed (GPU-0 gather OOM on T4x2).\n"
    "        # Whole-net DDP wrap happens in train.py main() after construction.\n"
    "        pass  # self.model.multi_gpu() removed\n"
)

_DDP_WRITER_OLD = (
    "        if not self.args.evaluate:\n"
    "            self.writer = SummaryWriter(self.args.checkpoint+'/'+'ckpt')\n"
)
_DDP_WRITER_NEW = (
    "        if not self.args.evaluate:\n"
    "            # DDP: only rank 0 writes TensorBoard (shared dir would corrupt).\n"
    "            _wrank = int(os.environ.get('RANK', '0'))\n"
    "            try:\n"
    "                if torch.distributed.is_available() and torch.distributed.is_initialized():\n"
    "                    _wrank = torch.distributed.get_rank()\n"
    "            except Exception:\n"
    "                pass\n"
    "            self.writer = SummaryWriter(self.args.checkpoint+'/'+'ckpt') if _wrank == 0 else None\n"
)

_DDP_RECORD_OLD = (
    "    def record(self,k,v,epoch):\n"
    "        self.writer.add_scalar(k, v, epoch)\n"
    "\n"
    "    def flush(self):\n"
    "        self.writer.flush()\n"
    "        sys.stdout.flush()\n"
)
_DDP_RECORD_NEW = (
    "    def record(self,k,v,epoch):\n"
    "        if getattr(self, 'writer', None) is None:\n"
    "            return\n"
    "        self.writer.add_scalar(k, v, epoch)\n"
    "\n"
    "    def flush(self):\n"
    "        if getattr(self, 'writer', None) is not None:\n"
    "            self.writer.flush()\n"
    "        sys.stdout.flush()\n"
)

_DDP_CLEAN_OLD = (
    "    def clean(self):\n"
    "        self.writer.close()\n"
)
_DDP_CLEAN_NEW = (
    "    def clean(self):\n"
    "        if getattr(self, 'writer', None) is not None:\n"
    "            self.writer.close()\n"
)

_DDP_SAVE_OLD = (
    "        state = {\n"
    "                    'epoch': self.current_epoch + 1,\n"
    "                    'nets': self.args.nets,\n"
    "                    'state_dict': self.model.state_dict(),\n"
)
_DDP_SAVE_NEW = (
    "        # DDP: unwrap so checkpoints stay single-GPU compatible (no `module.` prefix).\n"
    "        try:\n"
    "            from torch.nn.parallel import DistributedDataParallel as _DDP\n"
    "            _net = self.model.module if isinstance(self.model, _DDP) else self.model\n"
    "        except Exception:\n"
    "            _net = self.model\n"
    "        state = {\n"
    "                    'epoch': self.current_epoch + 1,\n"
    "                    'nets': self.args.nets,\n"
    "                    'state_dict': _net.state_dict(),\n"
)

_DDP_VGG1_OLD = (
    "        self.vgg = VGG16FeatureExtractor().cuda()\n"
    "        self.criterion = nn.L1Loss().cuda() if not relative else l1_relative\n"
)
_DDP_VGG1_NEW = (
    "        _vdev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'\n"
    "        self.vgg = VGG16FeatureExtractor().to(_vdev)\n"
    "        self.criterion = nn.L1Loss().to(_vdev) if not relative else l1_relative\n"
)
_DDP_VGG2_OLD = (
    "            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()\n"
)
_DDP_VGG2_NEW = (
    "            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(_vdev)\n"
)

_DDP_SLBR_ZG_OLD = (
    "            outputs = self.model(self.norm(inputs))\n"
    "            self.model.zero_grad_all()\n"
)
_DDP_SLBR_ZG_NEW = (
    "            outputs = self.model(self.norm(inputs))\n"
    "            _snet = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model\n"
    "            _snet.zero_grad_all()\n"
)
_DDP_SLBR_STEP_OLD = (
    "            total_loss.backward()\n"
    "            self.model.step_all()\n"
)
_DDP_SLBR_STEP_NEW = (
    "            total_loss.backward()\n"
    "            _snet2 = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model\n"
    "            _snet2.step_all()\n"
)
_DDP_SLBR_IMG_OLD = (
    "                self.writer.add_image('Image', image_dis, current_index)\n"
)
_DDP_SLBR_IMG_NEW = (
    "                if getattr(self, 'writer', None) is not None:\n"
    "                    self.writer.add_image('Image', image_dis, current_index)\n"
)
# Rank-guarded prints: both ranks now validate/train, so unguarded prints
# would double the log (and Kaggle truncates long logs).
_DDP_SLBR_TPRINT_OLD = (
    "            if current_index % 100 == 0:\n"
    "                print(suffix)\n"
)
_DDP_SLBR_TPRINT_NEW = (
    "            if current_index % 100 == 0 and int(os.environ.get('RANK', '0')) == 0:\n"
    "                print(suffix)\n"
)
_DDP_SLBR_VPRINT_OLD = (
    "                if i%100 == 0:\n"
    "                    print(suffix)\n"
)
_DDP_SLBR_VPRINT_NEW = (
    "                if i%100 == 0 and int(os.environ.get('RANK', '0')) == 0:\n"
    "                    print(suffix)\n"
)
_DDP_SLBR_VTOT_OLD = (
    "        print(\"Total:\")\n"
    "        print(suffix)\n"
)
_DDP_SLBR_VTOT_NEW = (
    "        if int(os.environ.get('RANK', '0')) == 0:\n"
    "            print(\"Total:\")\n"
    "            print(suffix)\n"
)
_DDP_SLBR_VITER_OLD = (
    "        print(\"Iter:%s,losses:%s,PSNR:%.4f,SSIM:%.4f\"%(epoch, losses_meter.avg,psnr_meter.avg,ssim_meter.avg))\n"
)
_DDP_SLBR_VITER_NEW = (
    "        if int(os.environ.get('RANK', '0')) == 0:\n"
    "            print(\"Iter:%s,losses:%s,PSNR:%.4f,SSIM:%.4f\"%(epoch, losses_meter.avg,psnr_meter.avg,ssim_meter.avg))\n"
)

_DDP_OPT_OLD = "        parser.add_argument('--gpu_id',default='0',type=str)"
_DDP_OPT_NEW = (
    "        parser.add_argument('--gpu_id',default='0',type=str)\n"
    "        parser.add_argument('--local-rank', '--local_rank', default=0, type=int)"
)

_TRAIN_DDP_PY = """from __future__ import print_function, absolute_import

import argparse
import torch,time,os

# NOTE: benchmark=True enables cuDNN autotune, which can pick faulty kernels
# for odd shapes on new torch/CUDA (seen as CUDA misaligned-address on T4 +
# torch 2.x). Fixed input size here, so autotune buys little — keep OFF.
torch.backends.cudnn.benchmark = False

from src.utils.misc import save_checkpoint, adjust_learning_rate
import src.models as models

import datasets as datasets
from options import Options
import numpy as np

def _is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _is_main():
    return (not _is_dist()) or torch.distributed.get_rank() == 0

def main(args):
    # DDP init FIRST: sampler + per-rank device depend on it.
    dist = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if dist:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        torch.distributed.init_process_group(backend='nccl')
    args.seed = 1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.dataset = args.dataset.lower()
    if args.dataset == 'clwd':
        dataset_func = datasets.CLWDDataset
    elif args.dataset == 'lvw':
        dataset_func = datasets.LVWDataset
    else:
        raise ValueError("Not known dataset:\\t{}".format(args.dataset))

    train_dataset = dataset_func('train',args)
    val_dataset = dataset_func('val',args)
    if dist:
        from torch.utils.data import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.train_batch, shuffle=False,
            sampler=train_sampler, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.test_batch, shuffle=False,
            sampler=val_sampler, num_workers=args.workers, pin_memory=True)
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    lr = args.lr
    data_loaders = (train_loader,val_loader)

    model = models.__dict__[args.models](datasets=data_loaders, args=args)
    if dist:
        # Whole-net DDP replaces per-submodule DataParallel (GPU-0 gather OOM).
        model.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.model)
        from torch.nn.parallel import DistributedDataParallel as DDP
        model.model = DDP(model.model, device_ids=[torch.cuda.current_device()],
                          # SLBR forward leaves some params (mask-branch maps)
                          # without grad each iter -> reducer needs this.
                          find_unused_parameters=True)
    if _is_main():
        print('============================ Initization Finish && Training Start =============================================')

    for epoch in range(model.args.start_epoch, model.args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(data_loaders, model, epoch, lr, args)
        if _is_main():
            print('\\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
            model.record('lr',lr, epoch)
        model.train(epoch)
        # model.validate(epoch)
        if args.freq < 0:
            # DDP: BOTH ranks validate (keeps train/eval toggling + DDP/SyncBN
            # forward counts symmetric; eval-only-on-rank0 desyncs NCCL and
            # hangs the next epoch with a broadcast timeout). Only rank 0
            # flushes/saves.
            model.validate(epoch)
            if _is_main():
                model.flush()
                model.save_checkpoint()
            if dist:
                torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    if dist:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False  # see note at top of file
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    args = parser.parse_args()
    # Under torchrun the rank mapping is already set — do NOT overwrite it.
    if int(os.environ.get('WORLD_SIZE', '1')) <= 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    _main_rank = int(os.environ.get('RANK', '0')) == 0
    if _main_rank:
        print('==================================== WaterMark Removal =============================================')
        print('==> {:50}: {:<}'.format("Start Time",time.ctime(time.time())))
        print('==> {:50}: {:<}'.format("USE GPU",os.environ.get('CUDA_VISIBLE_DEVICES', '(ddp-managed)')))
        print('==================================== Stable Parameters =============================================')
        for arg in vars(args):
            if type(getattr(args, arg)) == type([]):
                if ','.join([ str(i) for i in getattr(args, arg)]) == ','.join([ str(i) for i in parser.get_default(arg)]):
                    print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
            else:
                if getattr(args, arg) == parser.get_default(arg):
                    print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
        print('==================================== Changed Parameters =============================================')
        for arg in vars(args):
            if type(getattr(args, arg)) == type([]):
                if ','.join([ str(i) for i in getattr(args, arg)]) != ','.join([ str(i) for i in parser.get_default(arg)]):
                    print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
            else:
                if getattr(args, arg) != parser.get_default(arg):
                    print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
        print('==================================== Start Init Model  ===============================================')
    main(args)
    if _main_rank:
        print('==================================== FINISH WITHOUT ERROR =============================================')
"""


def _patch_device_block(txt: str, new_block: str) -> str:
    """Replace the `self.device = ...` statement with new_block.

    Paren-balanced scan from the RHS, so it matches the local SLBR_CPU
    two-liner AND the upstream single-liner AND any whitespace variant.
    String literals are skipped while counting parens.
    """
    import re as _re
    _m = _re.search(r"(?m)^(?P<indent>[ \t]*)self\.device\s*=", txt)
    if _m is None:
        print("no `self.device =` assignment; lines mentioning device/cuda:")
        for _ln, _line in enumerate(txt.splitlines(), 1):
            if "device" in _line or "cuda" in _line:
                print(f"  L{_ln}: {_line.rstrip()}")
        raise AssertionError("no `self.device =` assignment found in BasicModel.py")
    _start = _m.start()
    _rhs = txt[_m.end():]
    _par = _rhs.find("(")
    assert _par != -1, "no paren in self.device assignment"
    _depth = 0
    _instr = None
    _k = _par
    while True:
        assert _k < len(_rhs), "unbalanced parens in self.device assignment"
        _c = _rhs[_k]
        if _instr is not None:
            if _c == _instr:
                _instr = None
        elif _c in ("'", '"'):
            _instr = _c
        elif _c == "(":
            _depth += 1
        elif _c == ")":
            _depth -= 1
            if _depth == 0:
                break
        _k += 1
    _end = _m.end() + _k + 1  # just past the closing paren; rest (newline+) kept
    return txt[:_start] + new_block.rstrip("\n") + txt[_end:]


def _patch_repo_ddp(repo: str) -> None:
    from pathlib import Path as _P
    _repo = _P(repo)
    # Device first: format-agnostic (local SLBR_CPU two-liner vs upstream).
    _bp = _repo / "src" / "models" / "BasicModel.py"
    _btxt = _bp.read_text()
    if "per-rank device" in _btxt:
        print(f"ddp ok (cached): {_bp.relative_to(_repo)} (device)")
    else:
        _bp.write_text(_patch_device_block(_btxt, _DDP_DEVICE_NEW))
        print(f"ddp patched {_bp.relative_to(_repo)} (device)")
    _pairs = [
        (_repo / "src" / "networks" / "resunet.py",
         _DDP_MULTIGPU_OLD, _DDP_MULTIGPU_NEW, "Whole-net DDP wrap"),
        (_repo / "src" / "models" / "BasicModel.py",
         _DDP_NODP_OLD, _DDP_NODP_NEW, "DataParallel removed"),
        (_repo / "src" / "models" / "BasicModel.py",
         _DDP_WRITER_OLD, _DDP_WRITER_NEW, "only rank 0 writes"),
        (_repo / "src" / "models" / "BasicModel.py",
         _DDP_RECORD_OLD, _DDP_RECORD_NEW, "is None:\n            return"),
        (_repo / "src" / "models" / "BasicModel.py",
         _DDP_CLEAN_OLD, _DDP_CLEAN_NEW, "is not None:\n            self.writer.close()"),
        (_repo / "src" / "models" / "BasicModel.py",
         _DDP_SAVE_OLD, _DDP_SAVE_NEW, "unwrap so checkpoints"),
        (_repo / "src" / "utils" / "losses.py",
         _DDP_VGG1_OLD, _DDP_VGG1_NEW, "VGG16FeatureExtractor().to(_vdev)"),
        (_repo / "src" / "utils" / "losses.py",
         _DDP_VGG2_OLD, _DDP_VGG2_NEW, "norm=True).to(_vdev)"),
        (_repo / "src" / "models" / "SLBR.py",
         _DDP_SLBR_ZG_OLD, _DDP_SLBR_ZG_NEW, "_snet.zero_grad_all"),
        (_repo / "src" / "models" / "SLBR.py",
         _DDP_SLBR_STEP_OLD, _DDP_SLBR_STEP_NEW, "_snet2.step_all"),
        (_repo / "src" / "models" / "SLBR.py",
         _DDP_SLBR_IMG_OLD, _DDP_SLBR_IMG_NEW, "writer', None"),
        (_repo / "src" / "models" / "SLBR.py",
         _DDP_SLBR_TPRINT_OLD, _DDP_SLBR_TPRINT_NEW, "current_index % 100 == 0 and int(os.environ"),
        (_repo / "src" / "models" / "SLBR.py",
         _DDP_SLBR_VPRINT_OLD, _DDP_SLBR_VPRINT_NEW, "i%100 == 0 and int(os.environ"),
        (_repo / "src" / "models" / "SLBR.py",
         _DDP_SLBR_VTOT_OLD, _DDP_SLBR_VTOT_NEW, "== 0:\n            print(\"Total:\")"),
        (_repo / "src" / "models" / "SLBR.py",
         _DDP_SLBR_VITER_OLD, _DDP_SLBR_VITER_NEW, "== 0:\n            print(\"Iter:"),
        (_repo / "options.py",
         _DDP_OPT_OLD, _DDP_OPT_NEW, "local-rank"),
    ]
    for _path, _old, _new, _marker in _pairs:
        _txt = _path.read_text()
        if _marker in _txt:
            print(f"ddp ok (cached): {_path.relative_to(_repo)}")
            continue
        if _old not in _txt:
            # Diagnostic dump: show actual file lines containing the pattern's
            # distinctive tokens, so an upstream-drift miss is a 1-shot fix.
            import re as _re2
            _toks = sorted(
                set(_t for _t in _re2.findall(r"[A-Za-z_][A-Za-z0-9_.]{5,}", _old)),
                key=len, reverse=True)[:6]
            print(f"DDP pattern MISS in {_path.relative_to(_repo)}; context:")
            _shown = 0
            for _ln, _line in enumerate(_txt.splitlines(), 1):
                if any(_t in _line for _t in _toks):
                    print(f"  L{_ln}: {_line.rstrip()}")
                    _shown += 1
                    if _shown >= 15:
                        break
            raise AssertionError(f"DDP pattern changed in {_path}: {_old!r}")
        # NOTE: strip the trailing newline on BOTH sides: _old constants end
        # with "\n" (consuming the line break); replacing with a newline-less
        # _new would otherwise join the next line onto the last patched line.
        _path.write_text(_txt.replace(_old.rstrip("\n"), _new.rstrip("\n")))
        print(f"ddp patched {_path.relative_to(_repo)}")
    _tp = _repo / "train.py"
    _ttxt = _tp.read_text()
    if "DistributedSampler" not in _ttxt:
        _tp.write_text(_TRAIN_DDP_PY)
        print("ddp patched train.py (DDP rewrite)")
    else:
        print("ddp ok (cached): train.py")

_SITECUSTOMIZE = (
    "# Auto-loaded on every `python` startup when repo root is on PYTHONPATH\n"
    "# (exported above). Backstops any missed\n"
    "# `from skimage.measure import compare_*`.\n"
    "try:\n"
    "    import skimage.measure as _m\n"
    "    import skimage.metrics as _t\n"
    "    import numpy as _np\n"
    "    def _infer(_a, _b, _r):\n"
    "        if _r is not None:\n"
    "            return _r\n"
    "        return 255.0 if (float(_np.asarray(_a).max(initial=0)) > 1.5 or float(_np.asarray(_b).max(initial=0)) > 1.5) else 1.0\n"
    "    if not hasattr(_m, 'compare_psnr'):\n"
    "        _m.compare_psnr = lambda a, b, data_range=None, **k: _t.peak_signal_noise_ratio(a, b, data_range=_infer(a, b, data_range), **k)\n"
    "    if not hasattr(_m, 'compare_ssim'):\n"
    "        def _cs(X, Y, data_range=None, multichannel=False, channel_axis=None, win_size=None, **k):\n"
    "            _ca = -1 if (channel_axis is None and multichannel) else channel_axis\n"
    "            if win_size is not None:\n"
    "                k.setdefault('win_size', win_size)\n"
    "            return _t.structural_similarity(X, Y, data_range=_infer(X, Y, data_range), channel_axis=_ca, **k)\n"
    "        _m.compare_ssim = _cs\n"
    "except Exception:\n"
    "    pass\n"
)


def _patch_repo_skimage(repo: str) -> None:
    from pathlib import Path as _P
    _repo = _P(repo)
    _targets = [
        (_repo / "src" / "models" / "BasicModel.py",
         "from skimage.measure import compare_psnr,compare_ssim",
         _SK_COMPAT_IMPORT, "skimage.metrics"),
        (_repo / "src" / "models" / "SLBR.py",
         "from skimage.measure import compare_psnr,compare_ssim",
         _SK_COMPAT_IMPORT, "skimage.metrics"),
        (_repo / "test.py",
         "from skimage.measure import compare_ssim as ssim",
         _SK_TEST_IMPORT, "skimage.metrics"),
        (_repo / "datasets" / "base_dataset.py",
         "from albumentations import HorizontalFlip, RandomResizedCrop, Compose, DualTransform\n"
         "import albumentations.augmentations.transforms as transforms",
         _ALBU_COMPAT_IMPORT, "import albumentations as transforms"),
        (_repo / "src" / "utils" / "losses.py",
         "        vgg16 = models.vgg16(pretrained=True)",
         _VGG_COMPAT, "VGG16_Weights"),
        (_repo / "datasets" / "base_dataset.py",
         _HCOMPOSE_COMPAT_OLD,
         _HCOMPOSE_COMPAT_NEW, "read-only"),
        # replaceAll: train.py sets this in 2 places (top + __main__).
        # cuDNN autotune can pick faulty kernels on new torch/CUDA (CUDA
        # misaligned-address on T4 + torch 2.x); fixed 256px input here,
        # so autotune buys little — correctness first.
        (_repo / "train.py",
         "torch.backends.cudnn.benchmark = True",
         "torch.backends.cudnn.benchmark = False", "cudnn.benchmark = False"),
        (_repo / "src" / "models" / "BasicModel.py",
         "        self.model.load_state_dict(current_checkpoint['state_dict'], strict=True)",
         _RESUME_REMAP, "remapped {} DataParallel"),
        (_repo / "src" / "models" / "BasicModel.py",
         _RESUME_EPOCH_OLD,
         _RESUME_EPOCH_NEW, "explicit --start-epoch is the ONLY way"),
    ]
    for _path, _old, _new, _marker in _targets:
        _txt = _path.read_text()
        if _marker in _txt:
            print(f"compat ok (cached): {_path.relative_to(_repo)}")
            continue
        assert _old in _txt, f"pattern changed in {_path}: {_old!r}"
        _path.write_text(_txt.replace(_old, _new.rstrip("\n")))
        print(f"patched {_path.relative_to(_repo)}")
    _sc = _repo / "sitecustomize.py"
    if not (_sc.exists() and "compare_psnr" in _sc.read_text()):
        _sc.write_text(_SITECUSTOMIZE)
        print("wrote sitecustomize.py")
    else:
        print("compat ok (cached): sitecustomize.py")


_patch_repo_skimage(REPO_DIR)
# DDP conversion must run after the compat shims (independent regions).
_patch_repo_ddp(REPO_DIR)

# In-process fallback too (visual-QA cell imports src.models directly).
import skimage.measure as _skm
import skimage.metrics as _skmet
for _n, _f in (("compare_psnr", _skmet.peak_signal_noise_ratio),
               ("compare_ssim", _skmet.structural_similarity)):
    if not hasattr(_skm, _n):
        setattr(_skm, _n, _f)
        print(f"shimmed (in-process) skimage.measure.{_n}")

# Subprocess-proof check: fresh interpreter must import cleanly BEFORE the
# 30-epoch run. Fails fast here instead of after hours of setup.
subprocess.run([sys.executable, "-c",
                "from skimage.measure import compare_psnr,compare_ssim;"
                "import src.models, datasets; print('compat probe ok')"],
               check=True, cwd=REPO_DIR)

if not MOCK_RUN:
    # Real runs build VGGLoss -> 528MB vgg16 download. Fail fast here
    # (Kaggle needs Settings > Internet ON) instead of mid-training.
    subprocess.run([sys.executable, "-c",
                    "from torchvision.models import vgg16, VGG16_Weights;"
                    "vgg16(weights=VGG16_Weights.DEFAULT); print('vgg16 weights ok')"],
                   check=True, cwd=REPO_DIR)

# NOTE: train.py is DDP-aware (rewritten by _patch_repo_ddp): under torchrun it
# ignores --gpu_id / CUDA_VISIBLE_DEVICES management and uses LOCAL_RANK for
# the per-rank device. Single-GPU `python train.py` still works as before.
# Vendored `pytorch_ssim` / `pytorch_iou` live in repo root; train.py
# expects repo root on sys.path (it runs as cwd). Mirrored above via chdir.

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Finetune: same arch/loss flags as scripts/train.sh, lower LR + fewer epochs.
# Writes to CKPT_DIR/RUN_NAME/{checkpoint.pth.tar,model_best.pth.tar}.
# If the run times out, re-run this notebook with the SAME config plus
# `--resume <latest checkpoint.pth.tar> --start-epoch <N>` semantics handled by
# setting RESUME_FROM below to the interrupted checkpoint (weights reload, fresh
# optimizer at current LR schedule step).
RESUME_FROM = ""  # e.g. "/kaggle/working/slbr_ckpt/markless_slbr/checkpoint.pth.tar" to continue

import shlex

# MOCK_RUN trains from scratch (clwd_ckpt=None) and zeroes the VGG lambdas
# so VGGLoss never constructs (it would download vgg16 weights).
start_ckpt = RESUME_FROM or (str(clwd_ckpt) if clwd_ckpt else "")
_mock_lc = _mock_ls = "0" if MOCK_RUN else "0.25"


def _ckpt_epoch(path: str) -> str:
    """Read 'epoch' out of a checkpoint for --start-epoch continuations."""
    import torch as _torch_ckpt
    try:
        return str(int(_torch_ckpt.load(path, map_location="cpu",
                                        weights_only=True).get("epoch", 0) or 0))
    except Exception as _e:
        print(f"could not read epoch from {path}: {_e} — starting at 0")
        return "0"


# The patched resume() never adopts the checkpoint epoch (that made CLWD
# finetunes silently train zero epochs). Continuations resume explicitly.
start_epoch = _ckpt_epoch(start_ckpt) if RESUME_FROM else "0"
print(f"start_ckpt={start_ckpt or '(none)'}  start_epoch={start_epoch}")


def build_ddp_cmd(nproc: int) -> list:
    # DDP launch: one process per GPU. --train-batch/--test-batch are PER-GPU
    # (each rank loads this many); effective batch = per-GPU x nproc.
    _c = [
        sys.executable, "-u", "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={nproc}",
        "train.py",
        "--epochs", str(EPOCHS),
        "--schedule", *SCHEDULE.split(),
        "--lr", str(LR),
        "--beta1", str(BETA1), "--beta2", str(BETA2),
        "--checkpoint", CKPT_DIR,
        "--dataset_dir", DATA_CLWD,
        "--dataset", "clwd",
        "--nets", "slbr", "--models", "slbr",
        "--mask_mode", "res",
        "--k_center", "2", "--k_refine", "3", "--k_skip_stage", "3",
        "--use_refine",
        "--lambda_l1", "2",
        "--lambda_content", _mock_lc, "--lambda_style", _mock_ls,
        "--lambda_primary", "0.01", "--lambda_iou", "0.25",
        "--masked", "True",
        "--loss-type", "hybrid",
        "--train-batch", str(TRAIN_BATCH), "--test-batch", str(TEST_BATCH),
        "--workers", str(WORKERS),
        "--preprocess", "resize", "--input-size", "256", "--crop_size", "256",
        "--name", RUN_NAME,
        "--start-epoch", start_epoch,
    ]
    if start_ckpt:
        _c += ["--resume", start_ckpt]
    return _c


if not start_ckpt:
    print("no --resume (from scratch)")
cmd = build_ddp_cmd(N_GPU)
print(" ".join(shlex.quote(c) for c in cmd))
print(f"DDP: {N_GPU} ranks x {TRAIN_BATCH}/GPU = {TRAIN_BATCH * N_GPU} effective")
_rc = subprocess.run(cmd, check=False)  # streams repo progress bars to notebook log
if _rc.returncode != 0 and MOCK_RUN:
    # Same session, no extra queue slot: single rank + serialized kernels.
    # CUDA errors are async — the first traceback usually blames an innocent
    # later op; this rerun reports the TRUE failing line.
    print("\n===== attempt 1 failed — auto-debug rerun: "
          "nproc=1 + CUDA_LAUNCH_BLOCKING=1 =====")
    _dbg = build_ddp_cmd(1)
    print(" ".join(shlex.quote(c) for c in _dbg))
    _env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    _rc2 = subprocess.run(_dbg, check=False, env=_env)
    assert _rc2.returncode == 0, "debug rerun also failed — see TRUE traceback above"
    print("debug rerun PASSED on 1 rank + blocking")
if _rc.returncode != 0:
    raise subprocess.CalledProcessError(_rc.returncode, cmd)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Evaluate best checkpoint on the test(val) split (repo test.py metrics:
# PSNR/SSIM/RMSE/RMSEw + mask IoU/F1) + save 8 visual side-by-sides.
import json as _json

_run_dir = Path(CKPT_DIR) / RUN_NAME
_best = _run_dir / "model_best.pth.tar"
assert _best.exists(), f"missing {_best} — training may have been interrupted"
print(f"best: {_best} ({_best.stat().st_size / 1e6:.1f} MB)")

cmd = [sys.executable, "-u", "test.py",
       "--nets", "slbr", "--models", "slbr",
       "--input-size", "256", "--crop_size", "256",
       "--test-batch", "1", "--evaluate",
       "--dataset_dir", DATA_CLWD, "--dataset", "clwd",
       "--preprocess", "resize", "--no_flip",
       "--name", RUN_NAME, "--checkpoint", CKPT_DIR,
       "--mask_mode", "res",
       "--k_center", "2", "--k_refine", "3", "--k_skip_stage", "3",
       "--use_refine",
       "--resume", str(_best)]
print(" ".join(shlex.quote(c) for c in cmd))
subprocess.run(cmd, check=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Visual QA: J | GT | prev vs refinement — 8 test crops to PNG for eyeballing
# manhwa line-art recovery (repo test.py only prints numbers).
import cv2 as _cv2
import numpy as _np
import torch as _torch

from options import Options as _Options
import datasets as _datasets
import src.models as _models
import argparse as _argparse

_p = _Options().init(_argparse.ArgumentParser())
_args, _ = _p.parse_known_args(args=[
    "--nets", "slbr", "--models", "slbr", "--input-size", "256", "--crop_size", "256",
    "--test-batch", "1", "--dataset_dir", DATA_CLWD, "--dataset", "clwd",
    "--preprocess", "resize", "--no_flip", "--name", RUN_NAME, "--checkpoint", CKPT_DIR,
    "--mask_mode", "res", "--k_center", "2", "--k_refine", "3", "--k_skip_stage", "3",
    "--use_refine", "--resume", str(_best)])
# NOTE: single-GPU in-process eval (no DDP here — WORLD_SIZE unset in the
# notebook kernel, so the model stays unwrapped). Batch 1, save element [0].
_loader = _torch.utils.data.DataLoader(
    _datasets.CLWDDataset("val", _args), batch_size=1,
    shuffle=False, num_workers=0)
_machine = _models.__dict__[_args.models](datasets=(None, _loader), args=_args)
_machine.model.eval()
_prev = Path(_run_dir) / "preview"
_prev.mkdir(parents=True, exist_ok=True)
with _torch.no_grad():
    for _i, _b in enumerate(_loader):
        if _i >= 8:
            break
        _inp = _b["image"].to(_machine.device)
        _tgt = _b["target"].to(_machine.device)
        _out, _masks, _ = _machine.model(_machine.norm(_inp))
        _fin = _machine.denorm(_out[0] * _masks[0] + _machine.norm(_inp) * (1 - _masks[0]))
        for _tag, _t in (("J", _inp[0]), ("GT", _tgt[0]), ("pred", _fin[0])):
            _a = (_t.detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(_np.uint8)
            _cv2.imwrite(str(_prev / f"{_i:02d}_{_tag}.png"), _cv2.cvtColor(_a, _cv2.COLOR_RGB2BGR))
print(f"preview -> {_prev} ({len(list(_prev.glob('*.png')))} PNGs)")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Cleanup: Kaggle TRUNCATES notebook output past ~100k files — files beyond
# the limit are dropped. Only the repo .git objects + MOCK mock_clean dir are
# scaffolding here: DATA_CLWD points straight at the input/dataset root
# (read-only cache or mock dir), never a working-tree copy, so it must NOT
# be deleted. Checkpoints + preview are the only keepers.
# (Asserts keep us inside /kaggle/working.)
import shutil as _shutil


def _cleanup_working() -> None:
    _work = Path("/kaggle/working")
    _targets = [
        Path(REPO_DIR) / ".git",            # clone metadata, not needed in output
        Path("/kaggle/working/mock_clean"),  # MOCK_RUN synthetic source (tiny, but junk)
    ]
    for _t in _targets:
        assert _work in _t.parents, f"refusing to delete outside working: {_t}"
        if _t.is_symlink() or _t.is_file():
            _t.unlink()
            print(f"removed file {_t}")
        elif _t.is_dir():
            _n = sum(1 for _ in _t.rglob("*"))
            _shutil.rmtree(_t)
            print(f"removed dir {_t} ({_n} entries)")
        else:
            print(f"skip (missing): {_t}")
    for _p in _work.rglob("__pycache__"):
        if _p.is_dir():
            _shutil.rmtree(_p)
            print(f"removed {_p}")
    _left = sum(1 for _ in _work.rglob("*"))
    print(f"working tree entries left: {_left} (ckpt kept at {_run_dir})")


_cleanup_working()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Persist finetuned weights. Option A (explicit): kagglehub upload. Option B:
# "Save & Run All" (output capped at ~100k files — cleanup above keeps us under).
if MOCK_RUN:
    print(f"MOCK_RUN — dry run done, skipping upload (preview at {_run_dir})")
elif DO_UPLOAD:
    up = kagglehub.dataset_upload(
        UPLOAD_HANDLE, str(_run_dir),
        version_notes=f"slbr DDP finetune ep={EPOCHS} lr={LR} batch={TRAIN_BATCH}x{N_GPU} "
                      f"crops={counts['train']}+{counts['test']}")
    print("uploaded:", up)
else:
    print(f"DO_UPLOAD=False -> use Save & Run All to keep {_run_dir}")

"""Standalone synthetic watermark generator for MarklessMan (preview + bank).

NOT wired to generate_datasets.py / notebooks/kaggle_gen.py yet — this script exists so you
can eyeball the synthetic WM styles first (preview) and pre-generate a large
signature-unique bank (bank mode) for later 50/50 sampling without replacement.

Styles (from Shimu WM survey + scanlator banners):
  - url_strip:     1-line domain URL (5 templates, vertical variant, panel/
                   underline backdrops, shear/condense/tracking type xforms)
  - stacked_block: 2-4 line block (curated + composed CJK, mixed-script lines,
                   left/center/right align, tinted highlight bars, color rules)
  - number_badge:  badge 25 silhouettes (hexcut/concentric/fold/tornedge added),
                   procedural brands, 4 number formats, decorated/composed taglines
  - ribbon_bar:    translucent bar, 5 forms (rect/notch/split/pill/taper),
                   decorated taglines, jittered bar colors
  - pill_logo:     10 body treatments (notch/split added), procedural brands,
                   EP numbers / star suffixes
  - emblem_bar:    scanlator banner — emblem (15 families) + 9 bar shapes +
                   domain, ES/EN taglines, mascot/logo tail. Procedural brands,
                   emblem 0.45x-1.5x, floating/disc-less crescent-estar-bullseye.

Text: 16 fills + HSL jitter, contrasting/colored/double/thin outlines, 2- and
3-stop gradients (12 angles), hard/soft shadows (dx/dy to 6, blur to 4),
outer glow (10%), underline/strike bars (10%), faux-italic shear, condense,
tracking, per-char y-jitter, vertical + arc layouts.
Graphics: mascots (expanded palettes, striped shirts, scarves, glasses,
lean ±0.35, gaze ±0.18r), logos (19 families: torii/linked/speechmono added),
emblems with floating variants.
Degradation: per-sample alpha beta (35-95%), soften 0-2, log-uniform speckle
(0.0005-0.005, 20% clustered scratches), RGB grain (40%), ±1px edge
erode/dilate (30%), directional fade (15%).

Usage:
    py -3.13 synthetic_wm.py --out ./synth_wm_preview --num 60 --seed 7
    py -3.13 synthetic_wm.py --bank --bank-size 10000 --seed 7 --out ./synth_bank

Outputs:
    <out>/wm_XXXX.png          RGBA transparent watermark
    <out>/preview/pv_XXXX.jpg   watermark composited on gray for quick eyeball
    <out>/contact_sheet.jpg    grid of first-60 previews
    <out>/signatures.jsonl     (bank mode) per-sample visual signature
    <out>/bank_stats.json       (bank mode) uniqueness + style + alpha audit
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------- fonts
# malgun covers KR+latin but misses some CN glyphs (tofu); msyh covers CN+latin.
# Pick per-script so CJK lines never tofu.
# Linux (Kaggle) fallbacks appended after the Windows entries: nonexistent
# paths are skipped, so local Windows behavior is unchanged. DejaVu covers
# latin only — CJK lines still need Noto CJK; without it they fall back to
# PIL's default bitmap font (readable but ugly). If CJK quality matters on
# Kaggle, attach NotoSansCJK (.ttc) and add its path below.
_FONT_KR_FIRST = [
    r"C:\Windows\Fonts\malgunbd.ttf",   # Korean bold (covers latin + KR)
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\NotoSansKR-VF.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
_FONT_CN_FIRST = [
    r"C:\Windows\Fonts\msyhbd.ttc",     # Chinese bold (covers CN + latin + JP kana)
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\NotoSansKR-VF.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _needs_cn_font(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF:
            return True
    return False


def _load_from_list(cands: list[str], size: int) -> ImageFont.FreeTypeFont:
    key = (cands[0], size)
    if key in _font_cache:
        return _font_cache[key]
    for fp in cands:
        if Path(fp).exists():
            try:
                f = ImageFont.truetype(fp, size)
                _font_cache[key] = f
                return f
            except Exception:
                continue
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def load_font(size: int) -> ImageFont.FreeTypeFont:
    return _load_from_list(_FONT_KR_FIRST, max(8, int(size)))


def load_font_for(text: str, size: int) -> ImageFont.FreeTypeFont:
    size = max(8, int(size))
    cands = _FONT_CN_FIRST if _needs_cn_font(text) else _FONT_KR_FIRST
    return _load_from_list(cands, size)


def font_used_name(size: int) -> str:
    f = load_font(size)
    return getattr(f, "path", "default-bitmap")


# ---------------------------------------------------------------- vocab (FAKE names on purpose: generalization, not copies)
BRANDS = ["toonwave", "mangahub", "rawking", "noveltoon", "comicplus",
          "webtoonist", "manhwaraw", "toonsky", "mangadune", "rawchapter",
          "toonbarn", "mangafast", "webcomicfan", "dailymanhwa", "toonvault",
          "inkhaven", "panelpop", "stripesaga", "bubbleink", "frameflux",
          "inkdrift", "toonforge", "mangabloom", "rawhaven", "comicnest",
          "pixelfable", "inkport", "toonharbor", "mangamirth", "rawrelay"]
TLDS = [".com", ".net", ".to", ".cc", ".xyz", ".io", ".site",
        ".club", ".top", ".vip", ".app", ".art", ".lol", ".ink", ".pics"]
KR_LINES = ["가장 빠른 업데이트", "모든 웹툰 무료보기", "최신화 공유사이트",
            "고화질 만화 아카이브", "실시간 연재 중계", "무료 만화 뷰어",
            "매일 밤 12시 연재", "원본 화질로 감상", "회원가입 없이 보기",
            "인기 웹툰 몰아보기", "신작 알림 신청", "화요일 신작 공개"]
CN_LINES = ["最快更新漫画站", "免费在线观看", "高清漫画存档", "最新章节同步",
            "每日午夜连载", "无需注册观看", "原画质感欣赏", "人气漫画 binge",
            "新作提醒订阅", "周二新作公开"]
JP_LINES = ["無料漫画更新中", "最新話アーカイブ", "高画質RAWサイト",
            "毎日深夜連載", "登録不要で読む", "原画質で楽しむ",
            "人気漫画まとめ読み", "新作通知登録"]
ROMAN_LINES = ["MADE BY", "FASTEST UPDATE", "FREE COMICS", "RAW ARCHIVE", "DAILY UPDATE",
               "NEW CHAPTERS DAILY", "READ IN HD", "NO SIGN-UP NEEDED",
               "UPDATED AT MIDNIGHT", "WEEKLY DROPS", "UNCENSORED RAWS"]
# Dense-badge filler: short taglines + smallprint so large bodies never sit empty.
# Large badges stack 2-4 of these (tagline + info + smallprint paragraph).
BADGE_TAGLINES = ["DAILY UPDATE", "FREE PREVIEW", "RAW ARCHIVE", "FASTEST RELEASE",
                  "SUPPORT THE AUTHOR", "READ ON OFFICIAL SITE",
                  "NEW CHAPTERS DAILY", "HD RAWS INSIDE", "UPDATED AT MIDNIGHT",
                  "WEEKLY DROPS", "NO SIGN-UP NEEDED", "UNCENSORED EDITION"]
BADGE_SMALLPRINT = ["DO NOT RE-UPLOAD", "READ ON OFFICIAL SITE", "DAILY UPDATE",
                    "FREE PREVIEW ONLY", "SUPPORT THE AUTHOR",
                    "고화질 원본 보기", "무단 전재 금지", "최신화 매일 업데이트",
                    "高清原图请访问官网", "禁止转载", "最新章节每日更新",
                    "NO REPOST WITHOUT CREDIT", "REPORT LEAKS PLEASE",
                    "매일 밤 12시 연재", "원본 화질로 감상",
                    "无需注册观看", "每日午夜连载",
                    "登録不要で読む", "毎日深夜連載"]
# Scanlator-family vocab (fake brands — same shape as temple/omega/realm/utoon,
# never the real names). ES taglines cover the Spanish-speaking scanlator style.
SCAN_BRANDS = ["eclipses", "runescans", "voidscans", "haloscans", "onyxtoon",
               "riftcomic", "pyrescans", "lunarscans", "drifttoon", "crownscans",
               "emberhall", "tidescans", "novascan", "thorncomic", "zephyrtoon",
               "mistralink", "fenriscomic", "quartztoon", "sableink", "northpine",
               "cinderfox", "galewing", "mosscap", "ironlotus", "saltvault"]
ES_TAGLINES = ["LEÉ ANTES EN:", "ESCRIBE ESTE LINK EN TU NAVEGADOR",
               "LEE CON MEJOR CALIDAD EN:", "DISPONIBLE ANTES EN:",
               "CAPÍTULOS NUEVOS CADA DÍA:", "SIN REGISTRO:"]
EN_SCAN_TAGLINES = ["READ FIRST ON:", "READ WITH BETTER QUALITY AT:",
                    "EARLY RELEASES ON:", "SUPPORT US AT:",
                    "NEW CHAPTERS DAILY ON:", "NO SIGN-UP AT:"]
# CJK slot-fill composer: subject x verb x site -> hundreds of novel lines.
KR_SUBJ = ["최신화", "인기 웹툰", "신작 만화", "고화질 원본", "오늘의 연재"]
KR_VERB = ["매일 업데이트", "밤 12시 공개", "무료로 보기", "알림 신청", "몰아보기"]
KR_SITE = ["공유사이트", "아카이브", "뷰어", "중계방", "감상관"]
CN_SUBJ = ["最新章节", "人气漫画", "新作漫画", "高清原图", "今日连载"]
CN_VERB = ["每日更新", "午夜公开", "免费观看", "订阅提醒", "一口气看完"]
CN_SITE = ["分享站", "存档库", "观看器", "直播间", "欣赏馆"]
JP_SUBJ = ["最新話", "人気漫画", "新作漫画", "高画質原稿", "今日の連載"]
JP_VERB = ["毎日更新", "深夜公開", "無料で読む", "通知登録", "まとめ読み"]
JP_SITE = ["共有サイト", "アーカイブ", "ビューア", "配信部屋", "鑑賞館"]
# Decorator symbols sprinkled around domains/taglines (never in CJK body).
DECORATORS = ["★", "♥", "●", "◆", "▶", "×", "•", "※", "【", "】", "『", "』", "☆", "✦"]
# Procedural brand parts: syllable composer for near-infinite fake brands.
_BRAND_ONSETS = ["to", "ma", "ra", "co", "we", "in", "pa", "ba", "da", "ka",
                 "mi", "ru", "vo", "ha", "ze", "flu", "dri", "blo", "for", "nes"]
_BRAND_CORES = ["on", "an", "in", "un", "ar", "er", "or", "el", "is", "ow"]
_BRAND_SUFFIX = ["toon", "scans", "comic", "raw", "ink", "hub", "vault", "wave",
                 "nest", "flux", "haven", "relay", "pop", "saga", "port"]

FILLS = [(255, 255, 255), (20, 20, 20), (255, 122, 0), (235, 40, 40),
         (40, 160, 255), (255, 215, 0), (255, 90, 180), (60, 220, 130),
         (0, 180, 180), (140, 200, 40), (200, 60, 140), (150, 100, 235),
         (255, 170, 60), (40, 200, 140), (255, 235, 150), (170, 230, 180)]


def _procedural_brand(rng: random.Random) -> str:
    """Compose a novel fake brand: onset+core+suffix, e.g. 'drianelay'."""
    return (rng.choice(_BRAND_ONSETS) + rng.choice(_BRAND_CORES)
            + rng.choice(_BRAND_SUFFIX))


def _compose_cjk(rng: random.Random, kind: str) -> str:
    if kind == "kr":
        return f"{rng.choice(KR_SUBJ)} {rng.choice(KR_VERB)} {rng.choice(KR_SITE)}".strip()
    if kind == "cn":
        return f"{rng.choice(CN_SUBJ)}{rng.choice(CN_VERB)}{rng.choice(CN_SITE)}".strip()
    return f"{rng.choice(JP_SUBJ)}{rng.choice(JP_VERB)}{rng.choice(JP_SITE)}".strip()


def _decorate(text: str, rng: random.Random, p: float = 0.30) -> str:
    """Wrap/affix 0-2 decorator symbols around a latin/domain line."""
    if rng.random() > p:
        return text
    left = rng.choice(DECORATORS) if rng.random() < 0.7 else ""
    right = rng.choice(DECORATORS) if rng.random() < 0.7 else ""
    # brackets pair correctly
    pairs = {"【": "】", "『": "』"}
    if left in pairs:
        right = pairs[left]
    return f"{left}{text}{right}"


def _mixed_line(rng: random.Random) -> str:
    """Mixed-script line: CJK fragment + number/latin, mimics real hybrid WMs."""
    kind = rng.choice(["kr", "cn", "jp"])
    frag = _compose_cjk(rng, kind)
    tail = rng.choice([f" {rng.randint(1, 999)}", f" {rng.choice(ROMAN_LINES)}",
                       f"{rng.choice(TLDS)}", f" EP{rng.randint(1, 200)}"])
    return frag + tail


def fake_brand(rng: random.Random) -> str:
    if rng.random() < 0.45:
        return _procedural_brand(rng)
    return rng.choice(BRANDS)


def fake_domain(rng: random.Random) -> str:
    """5 URL templates (was 1): brand+num+TLD, subdomains, hyphen, short, caps."""
    brand = fake_brand(rng)
    num = rng.randint(1, 9999) if rng.random() < 0.3 else rng.randint(1, 999)
    tld = rng.choice(TLDS)
    t = rng.random()
    if t < 0.45:
        dom = f"{brand}{num}{tld}"
    elif t < 0.60:
        dom = f"{rng.choice(['www', 'm', 'raw', 'hd'])}.{brand}{num}{tld}"
    elif t < 0.72:
        dom = f"{brand}-{rng.choice(['raw', 'hd', 'hq', 'plus'])}{tld}"
    elif t < 0.84:
        dom = f"{brand}{tld}"  # short, no number
    elif t < 0.93:
        dom = f"{brand}{num}{tld}".upper()
    else:
        dom = f"{brand[:4]}{num}{tld}"  # truncated brand
    if rng.random() < 0.25:
        dom = _decorate(dom, rng, p=1.0)
    return dom


def pick_text_color(rng: random.Random) -> tuple[int, int, int]:
    # 65% HSL-jittered so the 16-entry table fans out across the bank.
    if rng.random() < 0.65:
        return _hsl_jitter(rng.choice(FILLS), rng)
    return rng.choice(FILLS)


def _hsl_jitter(rgb: tuple[int, int, int], rng: random.Random,
                dh: float = 12.0, ds: float = 0.15, dl: float = 0.12) -> tuple[int, int, int]:
    """Jitter an RGB fill in HLS space so the 16-entry FILLS table fans out."""
    import colorsys
    r, g, b = (c / 255.0 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + rng.uniform(-dh, dh) / 360.0) % 1.0
    l = min(0.95, max(0.05, l + rng.uniform(-dl, dl)))
    s = min(1.0, max(0.0, s + rng.uniform(-ds, ds)))
    if rng.random() < 0.12:  # occasional mute toward gray
        s *= rng.uniform(0.2, 0.5)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))


def jittered_fill(rng: random.Random) -> tuple[int, int, int]:
    base = rng.choice(FILLS)
    if rng.random() < 0.65:
        return _hsl_jitter(base, rng)
    return base


def outline_for(fill: tuple[int, int, int], rng: random.Random):
    """Contrasting outline + colored/double-stroke variants.

    Returns (stroke_rgb, width) or ((outer, inner), width) for double strokes.
    Callers that can't do double strokes should use _flat_stroke().
    """
    lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
    r = rng.random()
    w = max(1, rng.randint(1, 4)) if rng.random() < 0.15 else max(2, rng.randint(2, 4))
    if r < 0.55:
        if lum > 150:
            return (10, 10, 10), w
        return ((255, 255, 255) if rng.random() < 0.5 else (10, 10, 10)), w
    if r < 0.70:  # colored stroke
        return rng.choice([(235, 40, 40), (255, 215, 0), (40, 160, 255),
                           (20, 20, 20), (255, 255, 255)]), w
    if r < 0.80:  # double stroke marker
        outer = (10, 10, 10) if lum > 130 else (255, 255, 255)
        inner = rng.choice([(235, 40, 40), (255, 215, 0), (40, 160, 255)])
        return (outer, inner), w
    if r < 0.88:  # thin single px outline
        return ((10, 10, 10) if lum > 130 else (255, 255, 255)), 1
    return (10, 10, 10), w  # fallback dark


def _flat_stroke(sc, default=(10, 10, 10)):
    """Collapse a possibly-double (outer, inner) stroke to a single RGB."""
    if isinstance(sc, tuple) and len(sc) == 2 and isinstance(sc[0], tuple):
        return sc[0]
    return sc if sc is not None else default


# ------------------------------------------------- typography xforms
# Faux-italic (shear), faux-condensed (x-scale), tracking, per-char jitter,
# vertical stacking and arc-curve layout. All tile-based + seeded so the
# pregen bank stays deterministic per (seed, index).
def _roll_type_xform(rng: random.Random) -> dict:
    """Per-text-line geometric style: shear / condense / rotate / tracking."""
    return {
        "shear": rng.uniform(-0.22, 0.22) if rng.random() < 0.35 else 0.0,
        "scale_x": rng.uniform(0.80, 1.20) if rng.random() < 0.35 else 1.0,
        "rot": rng.uniform(-6, 6) if rng.random() < 0.25 else 0.0,
        "tracking": rng.randint(-1, 3) if rng.random() < 0.40 else 0,
        "yjit": rng.uniform(0, 2.0) if rng.random() < 0.30 else 0.0,
        "sizejit": rng.uniform(0, 0.15) if rng.random() < 0.25 else 0.0,
    }


def _xform_tile(tile: Image.Image, shear: float = 0.0, scale_x: float = 1.0,
                rot: float = 0.0) -> Image.Image:
    """Affine shear + x-scale + small rotation on an RGBA text tile."""
    w, h = tile.size
    if scale_x != 1.0:
        tile = tile.resize((max(1, int(w * scale_x)), h), Image.BICUBIC)
    if shear:
        # PIL AFFINE: x' = x + shear*y
        tile = tile.transform((tile.width + int(abs(shear) * tile.height) + 4, tile.height),
                              Image.AFFINE, (1, shear, -abs(shear) * tile.height / 2
                                             if shear < 0 else 0, 0, 1, 0),
                              resample=Image.BICUBIC)
    if rot:
        tile = tile.rotate(rot, expand=True, resample=Image.BICUBIC)
    return tile


def _render_line_tile(text: str, font, fill, sc, sw: int, rng: random.Random,
                      xf: dict | None = None) -> Image.Image:
    """Render one text line to its own RGBA tile with tracking + per-char jitter.

    Used when a type-xform rolled (shear/condense/rot). Plain lines keep the
    fast d.text path; this path costs more but breaks the uniform-glyph prior.
    """
    xf = xf or {}
    tracking = int(xf.get("tracking", 0) or 0)
    yjit = float(xf.get("yjit", 0) or 0)
    sizejit = float(xf.get("sizejit", 0) or 0)
    # measure total advance
    widths = []
    try:
        for ch in text:
            bb = font.getbbox(ch)
            widths.append(max(0, bb[2] - bb[0]) + tracking)
        bb0 = font.getbbox(text)
        asc = bb0[3] - bb0[1]
    except Exception:
        widths = [font.getlength(text)]
        asc = 20
    W = int(sum(widths)) + sw * 2 + 12
    H = int(asc + abs(yjit) * 2) + sw * 2 + 12
    tile = Image.new("RGBA", (max(1, W), max(1, H)), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    x = sw + 6
    yb = sw + 6 + abs(yjit)
    for ch, adv in zip(text, widths):
        try:
            cb = font.getbbox(ch)
            dy = rng.uniform(-yjit, yjit) if yjit else 0.0
            f4 = fill + (255,)
            if sw and sc is not None:
                s4 = _flat_stroke(sc) + (255,)
                td.text((x - cb[0], yb + dy - cb[1]), ch, font=font, fill=f4,
                        stroke_width=sw, stroke_fill=s4)
            else:
                td.text((x - cb[0], yb + dy - cb[1]), ch, font=font, fill=f4)
        except Exception:
            pass
        x += adv
    tile = _xform_tile(tile, shear=float(xf.get("shear", 0) or 0),
                       scale_x=float(xf.get("scale_x", 1.0) or 1.0),
                       rot=float(xf.get("rot", 0) or 0))
    _ = sizejit  # reserved: per-char size ladder lands in v2 (font re-load cost)
    return tile


def _render_vertical_tile(text: str, font, fill, sc, sw: int, rng: random.Random) -> Image.Image:
    """Top-to-bottom stacked characters (some real strip WMs read vertically)."""
    chars = list(text[:12])  # cap: vertical strips stay short
    Hs, Ws = [], []
    for ch in chars:
        try:
            bb = font.getbbox(ch)
            Ws.append(bb[2] - bb[0])
            Hs.append(bb[3] - bb[1])
        except Exception:
            Ws.append(10)
            Hs.append(14)
    W = max(Ws) + sw * 2 + 12 if Ws else 20
    H = sum(Hs) + len(chars) * 2 + sw * 2 + 12
    tile = Image.new("RGBA", (max(1, W), max(1, H)), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    y = sw + 6
    for ch, chh, chw in zip(chars, Hs, Ws):
        try:
            cb = font.getbbox(ch)
            x = (W - chw) / 2 - cb[0]
            f4 = fill + (255,)
            if sw and sc is not None:
                td.text((x, y - cb[1]), ch, font=font, fill=f4,
                        stroke_width=sw, stroke_fill=_flat_stroke(sc) + (255,))
            else:
                td.text((x, y - cb[1]), ch, font=font, fill=f4)
        except Exception:
            pass
        y += chh + 2
    return tile


def _render_arc_tile(text: str, font, fill, sc, sw: int, rng: random.Random) -> Image.Image:
    """Gentle arc layout: chars rotated along a circle segment (badge seals)."""
    n = max(1, len(text))
    arc = rng.uniform(40, 120)  # total sweep degrees
    radius = rng.uniform(60, 160)
    tiles = []
    for ch in text:
        try:
            bb = font.getbbox(ch)
            w, h = bb[2] - bb[0] + sw * 2 + 8, bb[3] - bb[1] + sw * 2 + 8
        except Exception:
            w, h = 16, 20
        t = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
        td = ImageDraw.Draw(t)
        try:
            bb = font.getbbox(ch)
            f4 = fill + (255,)
            if sw and sc is not None:
                td.text((sw + 4 - bb[0], sw + 4 - bb[1]), ch, font=font, fill=f4,
                        stroke_width=sw, stroke_fill=_flat_stroke(sc) + (255,))
            else:
                td.text((sw + 4 - bb[0], sw + 4 - bb[1]), ch, font=font, fill=f4)
        except Exception:
            pass
        tiles.append(t)
    W = int(radius * 2 + max(t.width for t in tiles) * 2 + 20)
    H = int(radius * 2 + max(t.height for t in tiles) * 2 + 20)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cx, cy = W / 2, H / 2
    a0 = -90 - arc / 2 + rng.uniform(-15, 15)
    for i, t in enumerate(tiles):
        a = math.radians(a0 + arc * i / max(1, n - 1))
        x = cx + radius * math.cos(a) - t.width / 2
        y = cy + radius * math.sin(a) - t.height / 2
        tr = t.rotate(math.degrees(a) + 90, expand=True, resample=Image.BICUBIC)
        _comp_tile_late(out, tr, int(x), int(y))
    # trim transparent margins
    bbox = out.getbbox()
    if bbox:
        out = out.crop((max(0, bbox[0] - 4), max(0, bbox[1] - 4),
                        min(W, bbox[2] + 4), min(H, bbox[3] + 4)))
    return out


# ---------------------------------------------------------------- helpers
def new_canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def draw_outlined_text(d: ImageDraw.ImageDraw, xy, text: str, font,
                       fill, stroke_fill, stroke_w: int):
    # PIL >= 8 supports stroke natively; keep a manual shadow pass for pop
    d.text(xy, text, font=font, fill=fill + (255,),
           stroke_width=stroke_w, stroke_fill=stroke_fill + (255,))


# ------------------------------------------------- text gradient + drop shadow
# Variants for synthetic text: 2/3-stop linear gradient fill (random angle)
# + hard-offset or soft-blurred drop shadow + outer glow + underline bar.
# Tile-based so gradient is masked by the glyph interior (stroke stays solid)
# and shadows can blur/offset without clipping. All rolls come from the
# caller's seeded rng; rendering happens at full opacity — apply_global_alpha
# scales it later.
TEXT_GRAD_P = 0.35
TEXT_SHADOW_P = 0.38
TEXT_SHADOW_SOFT_P = 0.5
TEXT_GLOW_P = 0.10
TEXT_UNDERLINE_P = 0.10
TEXT_GRAD_ANGLES = (90, 90, 0, 0, 45, -45, 30, -30, 60, -60, 15, -15)


def _norm_rgb(c) -> tuple[int, int, int]:
    return (int(c[0]), int(c[1]), int(c[2]))


def _pick_gradient_second(rng: random.Random, base) -> tuple[int, int, int]:
    base = _norm_rgb(base)
    r = rng.random()
    if r < 0.50:
        c2 = rng.choice(FILLS)
        if c2 == base:
            c2 = (255, 255, 255) if sum(base) < 380 else (10, 10, 10)
        return c2
    if r < 0.75:
        # tonal: darker shade or white tint of the base hue
        if rng.random() < 0.5:
            f = rng.uniform(0.45, 0.70)
            return (max(0, min(255, int(base[0] * f))),
                    max(0, min(255, int(base[1] * f))),
                    max(0, min(255, int(base[2] * f))))
        t = rng.uniform(0.35, 0.60)
        return (int(base[0] + (255 - base[0]) * t),
                int(base[1] + (255 - base[1]) * t),
                int(base[2] + (255 - base[2]) * t))
    # metallic-ish pole: base -> white or black
    if sum(base) < 380:
        return (245, 245, 245)
    return (10, 10, 10)


def _linear_gradient_rgb(w: int, h: int, c1, c2, angle_deg: float,
                         c3=None, mid: float = 0.5) -> Image.Image:
    """2/3-stop linear gradient RGB tile (no alpha). Angle in degrees:
    90 = top->bottom, 0 = left->right. c3=None -> 2-stop; else c1->c2->c3."""
    w, h = max(1, int(w)), max(1, int(h))
    c1, c2 = _norm_rgb(c1), _norm_rgb(c2)
    c3n = _norm_rgb(c3) if c3 is not None else None
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), -math.sin(a)
    # project pixel centers onto the direction, normalize to [0, 1]
    proj = ((np.arange(h, dtype=np.float32)[:, None] + 0.5 - h / 2.0) * dy
            + (np.arange(w, dtype=np.float32)[None, :] + 0.5 - w / 2.0) * dx)
    lo, hi = float(proj.min()), float(proj.max())
    t = np.zeros_like(proj) if hi <= lo else ((proj - lo) / (hi - lo))
    t = t[:, :, None]
    arr = np.empty((h, w, 3), dtype=np.uint8)
    if c3n is None:
        arr[:, :, 0] = (c1[0] + (c2[0] - c1[0]) * t[:, :, 0])
        arr[:, :, 1] = (c1[1] + (c2[1] - c1[1]) * t[:, :, 0])
        arr[:, :, 2] = (c1[2] + (c2[2] - c1[2]) * t[:, :, 0])
    else:
        m = float(mid)
        lo_m = np.clip(t[:, :, 0] / max(1e-6, m), 0, 1)
        hi_m = np.clip((t[:, :, 0] - m) / max(1e-6, 1 - m), 0, 1)
        for ch in range(3):
            arr[:, :, ch] = np.where(t[:, :, 0] < m,
                                    c1[ch] + (c2[ch] - c1[ch]) * lo_m,
                                    c2[ch] + (c3n[ch] - c2[ch]) * hi_m)
    return Image.fromarray(arr, "RGB")


def roll_text_fx(rng: random.Random, base_fill,
                 grad_p: float = TEXT_GRAD_P,
                 shadow_p: float = TEXT_SHADOW_P) -> dict:
    """Roll one consistent FX spec for a text element (share across chars)."""
    fx: dict = {"do_grad": False, "do_shadow": False,
                "do_glow": False, "do_underline": False}
    if rng.random() < grad_p:
        fx["do_grad"] = True
        fx["c1"] = _norm_rgb(base_fill)
        fx["c2"] = _pick_gradient_second(rng, base_fill)
        fx["angle"] = rng.choice(TEXT_GRAD_ANGLES)
        if rng.random() < 0.25:  # 3-stop metallic/pop gradient
            fx["c3"] = _pick_gradient_second(rng, fx["c2"])
            fx["mid"] = rng.uniform(0.35, 0.65)
        else:
            fx["c3"] = None
    if rng.random() < shadow_p:
        fx["do_shadow"] = True
        if rng.random() < TEXT_SHADOW_SOFT_P:
            fx["dx"], fx["dy"] = rng.randint(1, 6), rng.randint(1, 6)
            fx["blur"] = rng.choice([1, 1, 2, 3, 4])
            fx["opacity"] = rng.uniform(0.25, 0.60)
        else:
            fx["dx"], fx["dy"] = rng.randint(1, 4), rng.randint(1, 4)
            fx["blur"] = 0
            fx["opacity"] = rng.uniform(0.40, 0.70)
    if rng.random() < TEXT_GLOW_P:
        fx["do_glow"] = True
        fx["glow_color"] = rng.choice([(255, 255, 255), (255, 215, 0),
                                       (120, 220, 255), (255, 150, 180)])
        fx["glow_blur"] = rng.choice([2, 3, 4, 5])
        fx["glow_op"] = rng.uniform(0.25, 0.50)
    if rng.random() < TEXT_UNDERLINE_P:
        fx["do_underline"] = True
        fx["ul_pos"] = rng.choice(["under", "strike"])
        fx["ul_h"] = rng.choice([2, 3, 3, 4])
    return fx


def draw_fancy_text(im: Image.Image, xy, text: str, font,
                    fill, sc, sw: int,
                    fx: dict | None = None, rng: random.Random | None = None,
                    grad_p: float = TEXT_GRAD_P,
                    shadow_p: float = TEXT_SHADOW_P):
    """Draw text with optional gradient fill + drop shadow + glow + underline.

    `xy` uses the same convention as the old d.text call-sites:
    (x - bbox[0], y - bbox[1]) so ink lands at (x, y). Gradient replaces
    only the glyph interior; stroke `sc`/`sw` stays solid (double-stroke
    tuples collapse via _flat_stroke for the PIL rim; inner color is painted
    as a 1px second rim when present). Shadow is a dark offset copy (hard)
    or blurred copy (soft) composited underneath.
    Falls back to plain d.text when neither effect rolls.
    """
    fill3 = _norm_rgb(fill)
    dbl_inner = None
    if isinstance(sc, tuple) and len(sc) == 2 and isinstance(sc[0], tuple):
        dbl_inner = _norm_rgb(sc[1])
        sc = sc[0]
    sc3 = None if sc is None else _norm_rgb(sc)
    sw = int(sw or 0)
    if fx is None:
        if rng is None:
            fx = {"do_grad": False, "do_shadow": False}
        else:
            fx = roll_text_fx(rng, fill3, grad_p, shadow_p)
    if (not fx.get("do_grad") and not fx.get("do_shadow")
            and not fx.get("do_glow") and not fx.get("do_underline")
            and dbl_inner is None):
        d = ImageDraw.Draw(im)
        f4 = fill3 + (255,)
        if sw and sc3 is not None:
            d.text(xy, text, font=font, fill=f4,
                   stroke_width=sw, stroke_fill=sc3 + (255,))
        else:
            d.text(xy, text, font=font, fill=f4)
        return
    try:
        bb = font.getbbox(text)
    except Exception:
        d = ImageDraw.Draw(im)
        d.text(xy, text, font=font, fill=fill3 + (255,))
        return
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    if bw <= 0 or bh <= 0:
        return
    # guardrail: huge tiles (long ribbons) still fine, but cap runaway sizes
    if bw * bh > 4_000_000:
        d = ImageDraw.Draw(im)
        d.text(xy, text, font=font, fill=fill3 + (255,))
        return
    pad = sw + 4
    tw, th = bw + pad * 2, bh + pad * 2
    tx, ty = pad - bb[0], pad - bb[1]
    # --- stroke silhouette (solid stroke color so rim survives gradient)
    if sw and sc3 is not None:
        stroke_layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        sd = ImageDraw.Draw(stroke_layer)
        sd.text((tx, ty), text, font=font, fill=sc3 + (255,),
                stroke_width=sw, stroke_fill=sc3 + (255,))
        if dbl_inner is not None and sw >= 2:
            # inner 1px rim in accent color over the outer rim
            sd.text((tx, ty), text, font=font, fill=dbl_inner + (255,),
                    stroke_width=max(1, sw - 1), stroke_fill=dbl_inner + (255,))
    else:
        stroke_layer = None
    # --- interior mask (fill only, antialiased)
    mask = Image.new("L", (tw, th), 0)
    md = ImageDraw.Draw(mask)
    md.text((tx, ty), text, font=font, fill=255)
    # --- interior: gradient or solid
    if fx.get("do_grad"):
        c1 = _norm_rgb(fx.get("c1", fill3))
        c2 = _norm_rgb(fx.get("c2", fill3))
        c3 = fx.get("c3", None)
        grad = _linear_gradient_rgb(tw, th, c1, c2, fx.get("angle", 90),
                                    c3, fx.get("mid", 0.5)).convert("RGBA")
        grad.putalpha(mask)
        interior = grad
    else:
        interior = Image.new("RGBA", (tw, th), fill3 + (255,))
        interior.putalpha(mask)
    if stroke_layer is not None:
        out = Image.alpha_composite(stroke_layer, interior)
    else:
        out = interior
    # tile origin on dest: ink bbox lands at (X, Y) = (xy + bb0, xy + bb1)
    try:
        ox = int(xy[0] + bb[0] - pad)
        oy = int(xy[1] + bb[1] - pad)
    except Exception:
        ox, oy = int(xy[0]) - pad, int(xy[1]) - pad
    if fx.get("do_glow"):
        gc = _norm_rgb(fx.get("glow_color", (255, 255, 255)))
        gb = int(fx.get("glow_blur", 3) or 3)
        gop = float(fx.get("glow_op", 0.35))
        ga = out.getchannel("A").filter(ImageFilter.GaussianBlur(gb))
        ga = ga.point(lambda v: int(v * gop))
        glow = Image.new("RGBA", out.size, gc + (255,))
        glow.putalpha(ga)
        _comp_tile_late(im, glow, ox, oy)
    if fx.get("do_shadow"):
        dx, dy = int(fx.get("dx", 2)), int(fx.get("dy", 2))
        blur = int(fx.get("blur", 0) or 0)
        op = float(fx.get("opacity", 0.5))
        a = out.getchannel("A")
        if blur > 0:
            a = a.filter(ImageFilter.GaussianBlur(blur))
        sa = a.point(lambda v: int(v * op))
        shadow = Image.new("RGBA", out.size, (10, 10, 10, 255))
        shadow.putalpha(sa)
        _comp_tile_late(im, shadow, ox + dx, oy + dy)
    _comp_tile_late(im, out, ox, oy)
    if fx.get("do_underline"):
        try:
            ulh = int(fx.get("ul_h", 3))
            ink = fill3 + (255,)
            d2 = ImageDraw.Draw(im)
            # underline sits just under the ink bbox; strike crosses mid-ink
            base_y = oy + pad + bh if fx.get("ul_pos") == "under" else oy + pad + bh // 2
            x0 = ox + pad - 2
            x1 = ox + pad + bw + 2
            d2.rectangle([x0, base_y, x1, base_y + ulh], fill=ink)
        except Exception:
            pass


def _comp_tile_late(dst: Image.Image, src: Image.Image, dx: int, dy: int):
    """Clipped alpha_composite (same as _comp_tile, defined later in file)."""
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(dst.width, dx + src.width), min(dst.height, dy + src.height)
    if x1 <= x0 or y1 <= y0:
        return
    dst.alpha_composite(src.crop((x0 - dx, y0 - dy, x1 - dx, y1 - dy)), (x0, y0))


def apply_global_alpha(im: Image.Image, rng: random.Random,
                       lo: float | None = None, hi: float | None = None,
                       mode: float | None = None, conc: float | None = None) -> Image.Image:
    """Scale alpha so max alpha lands in ~35-95% (mode near 80%).

    Beta rescaled to [lo, hi]; None rolls per-sample so the bank spans faint
    ghosts through near-opaque stamps instead of one tight beta peak.
    """
    lo = rng.uniform(0.35, 0.45) if lo is None else lo
    hi = rng.uniform(0.85, 0.95) if hi is None else hi
    mode = rng.uniform(0.70, 0.85) if mode is None else mode
    conc = rng.uniform(6.0, 10.0) if conc is None else conc
    um = (mode - lo) / max(1e-6, hi - lo)
    um = min(0.95, max(0.05, um))
    k = lo + (hi - lo) * rng.betavariate(1 + conc * um, 1 + conc * (1 - um))
    a = im.getchannel("A")
    a = a.point(lambda v: int(v * k))
    im.putalpha(a)
    return im


def soften_alpha(im: Image.Image, rng: random.Random) -> Image.Image:
    r = rng.choices([0, 1, 2], weights=[40, 45, 15])[0]
    if r:
        a = im.getchannel("A").filter(ImageFilter.GaussianBlur(r))
        im.putalpha(a)
    return im


def add_speckle(im: Image.Image, np_rng: np.random.Generator, rng: random.Random | None = None,
                frac: float | None = None) -> Image.Image:
    """Faint print-texture dimming on a few alpha pixels.

    60% of samples skip entirely; otherwise frac is log-uniform 0.00005-0.0004
    (~10-20x fewer than the old 0.0005-0.005) and pixels are DIMMED to 30-60%
    instead of knocked to 0, so they read as print texture — never as
    scattered pinholes. 15% of the active rolls are short clustered
    scratches (3-8px) instead of uniform specks.
    """
    import math as _m2
    roll = rng.random() if rng is not None else float(np_rng.random())
    if frac is None and roll < 0.60:
        return im
    if frac is None:
        u = float(np_rng.random())
        frac = _m2.exp(_m2.log(0.00005) + u * (_m2.log(0.0004) - _m2.log(0.00005)))
    a = np.asarray(im.getchannel("A")).astype(np.int16)
    n = int(a.size * frac)
    if n <= 0:
        return im
    dim = int(np_rng.integers(76, 154))  # dim to ~30-60%, not 0
    clustered = ((rng.random() < 0.15) if rng is not None else False)
    if clustered:
        for _ in range(max(1, n // 6)):
            cy, cx = int(np_rng.integers(0, a.shape[0])), int(np_rng.integers(0, a.shape[1]))
            ln = int(np_rng.integers(3, 9))
            ang = float(np_rng.random() * 6.283)
            for k in range(ln):
                yy = min(a.shape[0] - 1, max(0, cy + int(k * _m2.sin(ang))))
                xx = min(a.shape[1] - 1, max(0, cx + int(k * _m2.cos(ang))))
                a[yy, xx] = min(a[yy, xx], dim)
    else:
        ys = np_rng.integers(0, a.shape[0], size=n)
        xs = np_rng.integers(0, a.shape[1], size=n)
        a[ys, xs] = np.minimum(a[ys, xs], dim)
    im.putalpha(Image.fromarray(a.clip(0, 255).astype(np.uint8)))
    return im


def add_grain(im: Image.Image, np_rng: np.random.Generator, rng: random.Random,
              amt: float | None = None) -> Image.Image:
    """Gaussian RGB grain under the alpha mask (print/scan noise). 40% roll."""
    if rng.random() > 0.40:
        return im
    amt = rng.uniform(3.0, 8.0) if amt is None else amt
    arr = np.asarray(im).astype(np.int16)
    noise = np_rng.normal(0, amt, size=arr.shape[:2])[..., None]
    m = (arr[..., 3:4] > 10).astype(np.int16)
    arr[..., :3] = np.clip(arr[..., :3] + noise * m, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def erode_alpha(im: Image.Image, rng: random.Random) -> Image.Image:
    """±1px alpha edge erosion/dilation (30%): breaks the pixel-perfect rim."""
    if rng.random() > 0.30:
        return im
    try:
        from PIL import ImageFilter as _IF
        a = im.getchannel("A")
        if rng.random() < 0.5:
            a = a.filter(_IF.MinFilter(3))
        else:
            a = a.filter(_IF.MaxFilter(3))
        im.putalpha(a)
    except Exception:
        pass
    return im


def fade_alpha_edge(im: Image.Image, rng: random.Random) -> Image.Image:
    """Directional alpha fade (15%): one edge dissolves like a cheap blend."""
    if rng.random() > 0.15:
        return im
    w, h = im.size
    edge = rng.choice(["left", "right", "top", "bottom"])
    span = rng.uniform(0.15, 0.40)
    a = np.asarray(im.getchannel("A")).astype(np.float32)
    if edge == "left":
        ramp = np.clip(np.arange(w)[None, :] / max(1, w * span), 0, 1)
    elif edge == "right":
        ramp = np.clip((w - np.arange(w))[None, :] / max(1, w * span), 0, 1)
    elif edge == "top":
        ramp = np.clip(np.arange(h)[:, None] / max(1, h * span), 0, 1)
    else:
        ramp = np.clip((h - np.arange(h))[:, None] / max(1, h * span), 0, 1)
    a = (a * np.broadcast_to(ramp, a.shape)).astype(np.uint8)
    im.putalpha(Image.fromarray(a))
    return im


def fit_font_size(text: str, max_w: int, start: int) -> ImageFont.FreeTypeFont:
    size = start
    while size > 10:
        f = load_font_for(text, size)
        bb = f.getbbox(text)
        if bb[2] - bb[0] <= max_w:
            return f
        size -= 2
    return load_font_for(text, 10)


# ---------------------------------------------------------------- styles
def gen_url_strip(rng: random.Random, np_rng) -> Image.Image:
    u = rng.random()
    if u < 0.30:
        url = f"http://{fake_domain(rng)}"
    elif u < 0.45:
        url = f"https://{fake_domain(rng)}"
    elif u < 0.55:
        url = f"WWW.{fake_domain(rng).upper()}"
    else:
        url = fake_domain(rng)
    if rng.random() < 0.12:  # vertical strip variant (some real WMs read top-down)
        fs = rng.randint(22, 40)
        font = load_font_for(url, fs)
        fill = pick_text_color(rng)
        sc, sw = outline_for(fill, rng)
        tile = _render_vertical_tile(url, font, fill, sc if sw else None, sw, rng)
        tile = _xform_tile(tile, rot=rng.uniform(-4, 4) if rng.random() < 0.3 else 0.0)
        pad = rng.randint(8, 16)
        im, d = new_canvas(tile.width + pad * 2, tile.height + pad * 2)
        _comp_tile_late(im, tile, pad, pad)
        return im
    fs = rng.randint(22, 60)
    font = load_font_for(url, fs)
    bb = font.getbbox(url)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = rng.randint(8, 18)
    W, H = tw + pad * 2 + rng.randint(0, 60), th + pad * 2
    im, d = new_canvas(W, H)
    # backdrop variants: transparent (usual) / translucent panel / underline bar
    broll = rng.random()
    if broll < 0.15:
        panel = rng.choice([(10, 10, 10), (240, 240, 240), (200, 30, 30),
                            (30, 90, 200)]) + (rng.randint(60, 160),)
        d.rectangle([2, 2, W - 2, H - 2], fill=panel)
    elif broll < 0.25:
        d.rectangle([4, H - 6, W - 4, H - 3],
                    fill=rng.choice([(235, 40, 40), (255, 215, 0),
                                     (255, 255, 255)]) + (200,))
    fill = pick_text_color(rng)
    sc, sw = outline_for(fill, rng)
    xf = _roll_type_xform(rng)
    if xf["shear"] or xf["scale_x"] != 1.0 or xf["rot"] or xf["tracking"]:
        tile = _render_line_tile(url, font, fill, sc if sw else None, sw, rng, xf)
        # keep strip canvas but re-center the transformed tile
        _comp_tile_late(im, tile, (W - tile.width) // 2, (H - tile.height) // 2)
    else:
        draw_fancy_text(im, (pad - bb[0], pad - bb[1]), url, font, fill, sc, sw, rng=rng)
    return im


def gen_stacked_block(rng: random.Random, np_rng) -> Image.Image:
    n_lines = rng.randint(2, 4) if rng.random() < 0.25 else rng.randint(2, 3)
    kind = rng.choice(["kr", "kr", "cn", "cn", "jp", "mix"])
    lines = []
    if kind == "mix":
        lines = [_mixed_line(rng)]
    else:
        cjk_pool = KR_LINES if kind == "kr" else (CN_LINES if kind == "cn" else JP_LINES)
        # 40%: composed novel line instead of a fixed list entry
        if rng.random() < 0.40:
            lines = [_compose_cjk(rng, kind)]
        else:
            lines = [rng.choice(cjk_pool)]
    if n_lines >= 3:
        lines.append(rng.choice(ROMAN_LINES) if rng.random() < 0.7 else _mixed_line(rng))
    if n_lines >= 4:
        lines.append(rng.choice(BADGE_SMALLPRINT))
    url = f"HTTPS://{fake_domain(rng).upper()}" if rng.random() < 0.7 \
        else fake_domain(rng)
    lines.append(url)
    align = rng.choices(["center", "left", "right"], weights=[60, 25, 15])[0]
    fs = rng.randint(22, 48)
    max_w = rng.randint(420, 640)
    fonts = [fit_font_size(t, max_w, fs if i < len(lines) - 1 else max(18, fs - 6))
             for i, t in enumerate(lines)]
    widths, heights = [], []
    for t, f in zip(lines, fonts):
        bb = f.getbbox(t)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1])
    pad_x, gap = rng.randint(10, 20), rng.randint(2, 10)
    W = max(widths) + pad_x * 2
    H = sum(heights) + gap * (len(lines) - 1) + rng.randint(16, 32)
    im, d = new_canvas(W, H)

    # optional translucent highlight bar behind a random line (buzztoon mimic)
    bar_line = -1
    if rng.random() < 0.5:
        bar_line = rng.randrange(len(lines))
        y0 = 8 + sum(heights[:bar_line]) + gap * bar_line - 2
        bar_col = rng.choice([(255, 255, 255), (255, 215, 0), (255, 90, 180),
                              (120, 220, 255)]) + (rng.randint(90, 200),)
        bar = Image.new("RGBA", (W, heights[bar_line] + 6), bar_col)
        im.alpha_composite(bar, (0, y0))

    y = 8
    for idx, (t, f, th) in enumerate(zip(lines, fonts, heights)):
        bb = f.getbbox(t)
        if idx == bar_line:
            # bar behind -> force dark/colored fill so text stays readable
            fill = rng.choice([(20, 20, 20), (200, 30, 30), (30, 90, 200), (255, 122, 0)])
        else:
            fill = (255, 255, 255) if t == url and rng.random() < 0.5 else pick_text_color(rng)
        sc, sw = outline_for(fill, rng)
        if align == "center":
            x = (W - (bb[2] - bb[0])) // 2
        elif align == "left":
            x = pad_x - rng.randint(0, 6)
        else:
            x = W - pad_x - (bb[2] - bb[0]) + rng.randint(0, 6)
        xf = _roll_type_xform(rng)
        if xf["shear"] or xf["scale_x"] != 1.0 or xf["tracking"]:
            tile = _render_line_tile(t, f, fill, sc if sw else None, sw, rng, xf)
            _comp_tile_late(im, tile, int(x - 4), int(y - 4))
        else:
            draw_fancy_text(im, (x - bb[0], y - bb[1]), t, f, fill, sc, sw, rng=rng)
        y += th + gap
    # optional top rule lines (buzztoon has thin rules above/below)
    if rng.random() < 0.35:
        for yy in (4, H - 5):
            rc = rng.choice([(255, 255, 255, 200), (255, 215, 0, 200),
                             (235, 40, 40, 200)])
            d.line([(6, yy), (W - 6, yy)], fill=rc, width=rng.choice([1, 2, 2]))
    return im


# ---------------------------------------------------------------- randomized mascot
SKINS = [(255, 224, 200), (245, 203, 175), (232, 184, 150), (255, 236, 220),
         (210, 160, 130), (245, 215, 190), (225, 195, 170), (255, 210, 180)]
HAIRS = [(30, 30, 30), (90, 55, 30), (220, 180, 80), (240, 150, 180),
         (90, 140, 255), (150, 90, 220), (80, 200, 130), (220, 70, 70),
         (230, 230, 230), (120, 200, 220),
         (60, 40, 80), (200, 120, 40), (40, 160, 160), (180, 60, 120),
         (100, 180, 60), (70, 70, 120)]
EYES = [(40, 60, 120), (120, 60, 30), (30, 100, 90), (150, 40, 90), (30, 30, 30),
        (90, 40, 140), (20, 120, 120), (170, 90, 20)]
SHIRTS = [(235, 60, 60), (60, 120, 235), (90, 190, 120), (255, 170, 40),
          (170, 100, 235), (240, 240, 240), (40, 40, 40), (235, 120, 180),
          (30, 130, 130), (150, 60, 60), (90, 90, 140), (60, 150, 90)]
# Extra families so non-chibi archetypes don't all reuse hair/shirt hues.
FURS = [(120, 85, 55), (200, 170, 130), (150, 150, 150), (235, 235, 235),
        (60, 60, 70), (220, 150, 90), (170, 120, 80), (100, 140, 160)]
HOODS = [(150, 60, 70), (90, 70, 150), (60, 110, 90), (200, 120, 60),
         (70, 90, 160), (110, 110, 120), (180, 80, 120), (50, 60, 80)]
HELMS = [(70, 80, 95), (120, 130, 145), (60, 90, 70), (150, 100, 60),
         (90, 60, 110), (180, 180, 190), (50, 70, 120), (100, 60, 60)]
GLOWS = [(255, 220, 80), (120, 220, 255), (255, 120, 120), (240, 240, 240),
         (150, 255, 170)]
STROKES = [(15, 15, 15), (40, 30, 20), (20, 30, 60)]


def _m_desat(rgb: tuple[int, int, int], rng: random.Random, p: float = 0.3) -> tuple[int, int, int]:
    """Sometimes mute a picked color toward gray so not every mascot is candy-bright."""
    if rng.random() < p:
        g = int(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        k = rng.uniform(0.35, 0.6)
        return (int(rgb[0] + (g - rgb[0]) * k), int(rgb[1] + (g - rgb[1]) * k),
                int(rgb[2] + (g - rgb[2]) * k))
    return rgb


def _m_stroke(rng: random.Random, r: int) -> tuple[tuple[int, int, int, int], int]:
    """Varied outline color + weight (legacy always used near-black r//12)."""
    base = rng.choice(STROKES)
    if rng.random() < 0.6:
        base = (15, 15, 15)
    w_choice = rng.random()
    if w_choice < 0.3:
        lw = max(1, r // 16)
    elif w_choice < 0.75:
        lw = max(2, r // 12)
    else:
        lw = max(2, r // 9)
    return base + (255,), lw


def _m_shift(x: float, y: float, cx: float, cy: float, lean: float) -> float:
    """Horizontal shear for pose lean: breaks vertical mirror symmetry."""
    return x + lean * (y - cy)


def _m_ell(d: ImageDraw.ImageDraw, cx: float, cy: float, w: float, h: float,
           lean: float, fig_cy: float, **kw):
    """Axis-aligned ellipse whose center is sheared by the figure lean."""
    ex = cx + lean * (cy - fig_cy)
    d.ellipse([ex - w, cy - h, ex + w, cy + h], **kw)


def _m_poly(d: ImageDraw.ImageDraw, pts: list, lean: float, fig_cy: float, **kw):
    d.polygon([(x + lean * (y - fig_cy), y) for (x, y) in pts], **kw)


def _m_arc(d: ImageDraw.ImageDraw, cx: float, cy: float, w: float, h: float,
           lean: float, fig_cy: float, **kw):
    ex = cx + lean * (cy - fig_cy)
    d.arc([ex - w, cy - h, ex + w, cy + h], **kw)


def _m_line(d: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float,
            lean: float, fig_cy: float, **kw):
    d.line([(_m_shift(x0, y0, 0, fig_cy, lean), y0),
            (_m_shift(x1, y1, 0, fig_cy, lean), y1)], **kw)


def _blob_pts(cx: float, cy: float, rx: float, ry: float, n: int,
              jit: float, rng: random.Random) -> list:
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rng.uniform(-0.15, 0.15)
        rr = rng.uniform(1.0 - jit, 1.0 + jit)
        pts.append((cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a)))
    return pts


# ------------------------------------------------- shared badge-shape helpers
def _rect_inset(W: float, H: float, m: float = 5.0):
    return m, m, W - m, H - m


def _chamfer_rect_pts(W, H, rng, m: float = 5.0) -> list:
    """Sharp rect with 1-4 cut corners. 'Not rounded' family."""
    x0, y0, x1, y1 = _rect_inset(W, H, m)
    cut = rng.uniform(8, min(28, min(W, H) * 0.22))
    corners = [rng.random() < 0.85 for _ in range(4)]  # TL,TR,BR,BL
    if not any(corners):
        corners[rng.randrange(4)] = True
    pts: list = []
    # walk clockwise from top edge: each cut corner becomes a diagonal
    pts.append((x0 + (cut if corners[0] else 0), y0))
    pts.append((x1 - (cut if corners[1] else 0), y0))
    if corners[1]:
        pts.append((x1, y0 + cut))
    pts.append((x1, y1 - (cut if corners[2] else 0)))
    if corners[2]:
        pts.append((x1 - cut, y1))
    pts.append((x0 + (cut if corners[3] else 0), y1))
    if corners[3]:
        pts.append((x0, y1 - cut))
    pts.append((x0, y0 + (cut if corners[0] else 0)))
    return pts


def _taper_pts(W, H, rng, m: float = 5.0) -> list:
    """Trapezoid: top width != bottom width + optional sideways slant."""
    x0, y0, x1, y1 = _rect_inset(W, H, m)
    mode = rng.choice(["narrow_top", "narrow_bot", "narrow_both_mid"])
    inset = rng.uniform(0.08, 0.28) * (x1 - x0) / 2
    slant = rng.uniform(-0.15, 0.15) * (x1 - x0)
    if mode == "narrow_top":
        return [(x0 + inset + slant * 0.3, y0), (x1 - inset + slant * 0.3, y0),
                (x1, y1), (x0, y1)]
    if mode == "narrow_bot":
        return [(x0, y0), (x1, y0),
                (x1 - inset - slant * 0.3, y1), (x0 + inset - slant * 0.3, y1)]
    # bowed/hourglass-ish: pinch mid via 6-pt hexagon
    mid_in = rng.uniform(0.04, 0.12) * (x1 - x0) / 2
    my0, my1 = y0 + (y1 - y0) * 0.28, y0 + (y1 - y0) * 0.72
    return [(x0, y0), (x1, y0), (x1 - mid_in, my0),
            (x1 - mid_in, my1), (x1, y1), (x0, y1),
            (x0 + mid_in, my1), (x0 + mid_in, my0)]


def _slant_pts(W, H, rng, m: float = 5.0) -> list:
    """Parallelogram lean (italic badge)."""
    x0, y0, x1, y1 = _rect_inset(W, H, m)
    lean = rng.uniform(0.08, 0.24) * (x1 - x0) * rng.choice([-1, 1])
    return [(x0 + lean, y0), (x1 + lean, y0), (x1 - lean, y1), (x0 - lean, y1)]


def _asym_round_pts(W, H, rng, m: float = 5.0, seg: int = 5) -> list:
    """Per-corner rounded rect as polygon: mixes rounded + sharp corners."""
    x0, y0, x1, y1 = _rect_inset(W, H, m)
    maxr = min(W, H) * 0.28
    style = rng.choice(["one_sharp", "two_diag", "pill_side", "random"])
    if style == "one_sharp":
        radii = [rng.uniform(12, maxr), rng.uniform(12, maxr),
                 rng.uniform(12, maxr), 0.0]
        rng.shuffle(radii)
    elif style == "two_diag":
        r1, r2 = rng.uniform(14, maxr), rng.uniform(0, 8)
        radii = [r1, r2, r1, r2]
    elif style == "pill_side":
        r = min((x1 - x0), (y1 - y0)) / 2 - 1
        side = rng.choice(["left", "top"])
        # radii order: TL,TR,BR,BL — round one side, keep opposite sharp
        radii = [r, 0.0, 0.0, r] if side == "left" else [r, r, 0.0, 0.0]
    else:
        radii = [rng.choice([0.0, rng.uniform(6, maxr)]) for _ in range(4)]
    corners = [(x1 - radii[1], y0 + radii[1], -90, 0),    # TR
               (x1 - radii[2], y1 - radii[2], 0, 90),     # BR
               (x0 + radii[3], y1 - radii[3], 90, 180),   # BL
               (x0 + radii[0], y0 + radii[0], 180, 270)]  # TL
    # walk TL -> TR -> BR -> BL
    order = [corners[3], corners[0], corners[1], corners[2]]
    pts: list = []
    for cx, cy, a0, a1 in order:
        r = math.hypot(cx - (x0 if cx < (x0 + x1) / 2 else x1),
                       cy - (y0 if cy < (y0 + y1) / 2 else y1))
        r = abs(r)
        if r < 0.5:
            pts.append((x0 if cx < (x0 + x1) / 2 else x1,
                        y0 if cy < (y0 + y1) / 2 else y1))
            continue
        for i in range(seg + 1):
            a = math.radians(a0 + (a1 - a0) * i / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _wavy_rect_pts(W, H, rng, m: float = 6.0, n: int = 12) -> list:
    """Squiggly / hand-wobble rectangle: sine + random phase per edge."""
    x0, y0, x1, y1 = _rect_inset(W, H, m)
    amp = rng.uniform(3, min(10, min(W, H) * 0.07))
    f1, f2 = rng.randint(2, 5), rng.randint(2, 5)
    p1, p2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    jag = rng.uniform(0, 2.0)  # extra random jitter on top of sine
    pts: list = []
    for i in range(n + 1):  # top L->R
        t = i / n
        pts.append((x0 + (x1 - x0) * t,
                    y0 + amp * math.sin(t * f1 * math.pi + p1) + rng.uniform(-jag, jag)))
    for i in range(1, n + 1):  # right T->B
        t = i / n
        pts.append((x1 + amp * math.sin(t * f2 * math.pi + p2) + rng.uniform(-jag, jag),
                    y0 + (y1 - y0) * t))
    for i in range(1, n + 1):  # bottom R->L
        t = i / n
        pts.append((x1 - (x1 - x0) * t,
                    y1 + amp * math.sin(t * f1 * math.pi + p1 + 1.7) + rng.uniform(-jag, jag)))
    for i in range(1, n):  # left B->T (skip closing dup)
        t = i / n
        pts.append((x0 + amp * math.sin(t * f2 * math.pi + p2 + 3.1) + rng.uniform(-jag, jag),
                    y1 - (y1 - y0) * t))
    return pts


def _scallop_pts(cx, cy, rx, ry, rng, n: int | None = None) -> list:
    """Stamp / scalloped edge: alternating outer/inner radius."""
    n = n or rng.randint(10, 18)
    depth = rng.uniform(0.04, 0.09)
    rot = rng.uniform(0, 6.28)
    pts: list = []
    for i in range(n * 2):
        a = rot + math.pi * i / n
        rr = 1.0 if i % 2 == 0 else (1.0 - depth)
        pts.append((cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a)))
    return pts


def _seal_pts(cx, cy, rx, ry, rng) -> list:
    """Starburst splash seal with random teeth count + depth."""
    n = rng.randint(9, 18)
    depth = rng.uniform(0.12, 0.30)
    rot = rng.uniform(0, 6.28)
    pts: list = []
    for i in range(n * 2):
        a = rot + math.pi * i / n
        rr = 1.0 if i % 2 == 0 else (1.0 - depth)
        pts.append((cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a)))
    return pts


def _ticket_pts(W, H, rng, m: float = 5.0, seg: int = 8) -> list:
    """Rect with semicircle notches punched on left/right (transit ticket)."""
    x0, y0, x1, y1 = _rect_inset(W, H, m)
    nr = rng.uniform(8, min(15, H * 0.16))
    ny = (y0 + y1) / 2 + rng.uniform(-H * 0.1, H * 0.1)
    pts: list = [(x0, y0), (x1, y0), (x1, ny - nr)]
    for i in range(1, seg):  # right notch arcs inward
        a = -math.pi / 2 + math.pi * i / seg
        pts.append((x1 - nr + nr * math.cos(a), ny + nr * math.sin(a)))
    pts += [(x1, ny + nr), (x1, y1), (x0, y1), (x0, ny + nr)]
    for i in range(1, seg):  # left notch arcs inward
        a = math.pi / 2 + math.pi * i / seg
        pts.append((x0 + nr + nr * math.cos(a), ny + nr * math.sin(a)))
    pts.append((x0, ny - nr))
    return pts


def _bookmark_pts(W, H, rng, m: float = 5.0) -> list:
    """Ribbon bookmark: rect with V/U notch cut from bottom (or top)."""
    x0, y0, x1, y1 = _rect_inset(W, H, m)
    notch_d = rng.uniform(15, min(34, H * 0.25))
    notch_w = rng.uniform(0.18, 0.34) * (x1 - x0)
    cxm = (x0 + x1) / 2 + rng.uniform(-8, 8)
    flip = rng.random() < 0.25  # notch on top instead
    u_round = rng.random() < 0.4
    if not flip:
        if u_round:
            return [(x0, y0), (x1, y0), (x1, y1),
                    (cxm + notch_w, y1), (cxm + notch_w * 0.4, y1 - notch_d),
                    (cxm - notch_w * 0.4, y1 - notch_d), (cxm - notch_w, y1), (x0, y1)]
        return [(x0, y0), (x1, y0), (x1, y1),
                (cxm + notch_w, y1), (cxm, y1 - notch_d), (cxm - notch_w, y1), (x0, y1)]
    return [(x0, y0), (cxm - notch_w, y0), (cxm, y0 + notch_d),
            (cxm + notch_w, y0), (x1, y0), (x1, y1), (x0, y1)]


def _random_path_pts(cx, cy, rx, ry, rng, n: int | None = None) -> list:
    """Full random vector path: random angle gaps + random radii, closed."""
    n = n or rng.randint(7, 13)
    angs = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n))
    jit = rng.uniform(0.15, 0.38)
    sq = rng.choice([1.0, 1.0, 0.75])  # sometimes squash vertically
    return [(cx + rx * rng.uniform(1 - jit, 1 + jit * 0.6) * math.cos(a),
             cy + ry * sq * rng.uniform(1 - jit, 1 + jit * 0.6) * math.sin(a))
            for a in angs]


def _shear_pts(pts: list, rng, amt: float | None = None) -> list:
    """Orthogonal lean modifier: breaks vertical mirror symmetry."""
    if not pts:
        return pts
    cyy = sum(p[1] for p in pts) / len(pts)
    lean = amt if amt is not None else rng.uniform(-0.12, 0.12)
    return [(x + lean * (y - cyy), y) for (x, y) in pts]


def _roughen_pts(pts: list, rng, amt: float = 2.5) -> list:
    """Hand-drawn modifier: small per-vertex jitter."""
    return [(x + rng.uniform(-amt, amt), y + rng.uniform(-amt, amt)) for (x, y) in pts]


def _stroke_poly(d: ImageDraw.ImageDraw, pts: list, body, edge, close: bool = True):
    d.polygon(pts, fill=body, outline=edge)


def _head_poly_pts(cx: float, cy: float, rx: float, ry: float, jaw: str,
                   rng: random.Random) -> list:
    """Head outline with varied jaw: round / flat / pointed chin."""
    pts = []
    n = 26
    for i in range(n):
        a = 2 * math.pi * i / n
        px, py = math.cos(a), math.sin(a)
        if jaw == "flat" and py > 0.55:
            py = 0.55 + (py - 0.55) * 0.35
        elif jaw == "point" and py > 0.3:
            # taper sides toward chin, extend chin down
            taper = 1.0 - (py - 0.3) * 0.45
            px *= max(0.45, taper)
            if py > 0.9:
                py = 0.9 + (py - 0.9) * 1.6
        pts.append((cx + rx * px, cy + ry * py))
    return pts


# ---------------------------------------------------------------- mascot archetypes
def _mascot_chibi(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail):
    """Legacy chibi head, diversified: head aspect + jaw + fringe + one-sided extras."""
    skin = _m_desat(rng.choice(SKINS), rng) + (255,)
    hair = _m_desat(rng.choice(HAIRS), rng) + (255,)
    iris = rng.choice(EYES) + (255,)
    shirt = _m_desat(rng.choice(SHIRTS), rng) + (255,)
    wr = r * rng.uniform(0.85, 1.12)
    hr = r * rng.uniform(0.9, 1.1)
    jaw = rng.choice(["round", "round", "flat", "point"])
    style = rng.choice(["bob", "spiky", "long", "twintails", "none"])
    extra = rng.choice(["none", "none", "catears", "horns", "halo", "bow", "ahoge"])
    side = rng.choice([-1, 1])

    # --- back hair
    if style == "spiky":
        pts = []
        n = rng.randint(7, 10)
        for i in range(2 * n):
            rr = wr * (1.35 if i % 2 == 0 else 1.0)
            a = math.pi + i * math.pi / n
            pts.append((cx + rr * math.cos(a), cy + rr * 0.9 * math.sin(a) - r * 0.1))
        _m_poly(d, pts, lean, cy, fill=hair, outline=ln)
    elif style == "none":
        _m_ell(d, cx, cy - r * 0.15, wr * 1.02, hr * 1.02, lean, cy, fill=hair, outline=ln)
    else:
        _m_ell(d, cx, cy - r * 0.1, wr * 1.25, hr * 1.2, lean, cy, fill=hair, outline=ln)
    if style == "twintails":
        tr = r * rng.uniform(0.45, 0.6)
        for sx in (-1, 1):
            tx, tyy = cx + sx * r * 1.35, cy + r * 0.35
            _m_ell(d, tx, tyy, tr, tr * 1.6, lean, cy, fill=hair, outline=ln)

    # --- extras (one-sided bow/ahoge break mirror symmetry)
    if extra == "catears":
        for sx in (-1, 1):
            _m_poly(d, [(cx + sx * wr * 0.45, cy - hr * 0.75),
                        (cx + sx * wr * 0.95, cy - hr * 0.7),
                        (cx + sx * wr * 0.60, cy - hr * 1.45)],
                    lean, cy, fill=hair, outline=ln)
    elif extra == "horns":
        for sx in (-1, 1):
            _m_poly(d, [(cx + sx * wr * 0.5, cy - hr * 0.8),
                        (cx + sx * wr * 0.75, cy - hr * 0.8),
                        (cx + sx * wr * 0.45, cy - hr * 1.5)],
                    lean, cy, fill=(240, 240, 240, 255), outline=ln)
    elif extra == "halo":
        _m_ell(d, cx, cy - hr * 1.6, r * 0.7, r * 0.25, lean, cy,
               outline=(255, 215, 0, 255), width=max(2, r // 10))
    elif extra == "bow":
        bx = cx + side * wr * 0.75
        by = cy - hr * 0.95
        bs = r * 0.35
        bc = rng.choice([(235, 40, 40), (240, 150, 180), (150, 90, 255)]) + (255,)
        _m_poly(d, [(bx, by), (bx - side * bs, by - bs * 0.7),
                    (bx - side * bs, by + bs * 0.7)], lean, cy, fill=bc, outline=ln)
        _m_poly(d, [(bx, by), (bx + side * bs, by - bs * 0.7),
                    (bx + side * bs, by + bs * 0.7)], lean, cy, fill=bc, outline=ln)

    # --- head with jaw shape
    _m_poly(d, _head_poly_pts(cx, cy, wr, hr, jaw, rng), lean, cy, fill=skin, outline=ln)

    # --- fringe / cap variants
    fringe = rng.choice(["zigzag", "zigzag", "sidepart", "scallop", "beanie", "none"])
    if fringe == "zigzag":
        teeth = rng.randint(3, 5)
        top = cy - hr * 0.95
        pts = [(cx - wr * 0.95, top + hr * 0.35)]
        for i in range(teeth * 2 + 1):
            x = cx - wr * 0.95 + i * (wr * 1.9 / (teeth * 2))
            y = top if i % 2 == 0 else top + hr * 0.55
            pts.append((x, y))
        pts.append((cx + wr * 0.95, top + hr * 0.35))
        _m_poly(d, pts, lean, cy, fill=hair)
        _m_arc(d, cx, cy, wr, hr, lean, cy, start=200, end=340, fill=ln, width=lw)
    elif fringe == "sidepart":
        _m_poly(d, [(cx - wr * 0.95, cy - hr * 0.9), (cx + wr * 0.95, cy - hr * 0.9),
                    (cx + wr * 0.2, cy - hr * 0.1), (cx - wr * 0.95, cy - hr * 0.25)],
                lean, cy, fill=hair)
    elif fringe == "scallop":
        for i in range(3):
            sx = cx - wr * 0.6 + i * wr * 0.6
            d.pieslice([_m_shift(sx - wr * 0.35, cy - hr * 0.9, 0, cy, lean),
                        cy - hr * 1.0,
                        _m_shift(sx + wr * 0.35, cy - hr * 0.9, 0, cy, lean),
                        cy - hr * 0.15],
                       start=180, end=360, fill=hair)
    elif fringe == "beanie":
        cap = rng.choice([(235, 60, 60), (60, 120, 235), (90, 190, 120),
                          (255, 170, 40), (150, 90, 235)]) + (255,)
        d.pieslice([_m_shift(cx - wr * 1.0, cy - hr * 1.5, 0, cy, lean), cy - hr * 1.55,
                    _m_shift(cx + wr * 1.0, cy - hr * 1.5, 0, cy, lean), cy + hr * 0.1],
                   start=180, end=360, fill=cap, outline=ln)
        d.rectangle([_m_shift(cx - wr * 1.0, cy - hr * 0.55, 0, cy, lean), cy - hr * 0.55,
                     _m_shift(cx + wr * 1.0, cy - hr * 0.55, 0, cy, lean), cy - hr * 0.2],
                    fill=cap, outline=ln)
    if extra == "ahoge":
        _m_arc(d, cx + side * r * 0.2, cy - hr * 1.5, r * 0.45, r * 0.6,
               lean, cy, start=280, end=60, fill=ln, width=max(2, r // 14))

    if style in ("long", "bob"):
        ll = r * (1.5 if style == "long" else 0.9)
        sides = (-1, 1) if rng.random() < 0.6 else (side,)
        for sx in sides:
            _m_poly(d, [(cx + sx * wr * 0.85, cy - hr * 0.2),
                        (cx + sx * wr * 1.05, cy - hr * 0.2),
                        (cx + sx * wr * 0.9, cy + ll)], lean, cy, fill=hair)

    # --- eyes with gaze + 3/4-view + size mismatch
    ex = wr * 0.36
    ey = cy + hr * 0.08
    qv = rng.choice([-1, 1]) * r * 0.12 if rng.random() < 0.25 else 0.0
    estyle = rng.choice(["round", "round", "oval", "happy", "slit"])
    if estyle == "happy":
        for sx in (-1, 1):
            shrink = 0.8 if (qv and sx != (1 if qv > 0 else -1)) else 1.0
            _m_arc(d, cx + sx * ex + qv, ey, r * 0.22 * shrink, r * 0.18,
                   lean, cy, start=180, end=360, fill=ln, width=max(2, r // 12))
    elif estyle == "slit":
        for sx in (-1, 1):
            shrink = 0.7 if (qv and sx != (1 if qv > 0 else -1)) else 1.0
            _m_ell(d, cx + sx * ex + qv, ey, r * 0.22 * shrink, r * 0.10,
                   lean, cy, fill=ln)
    else:
        ew, eh = (r * 0.26, r * 0.32) if estyle == "round" else (r * 0.20, r * 0.38)
        for sx in (-1, 1):
            shrink = rng.uniform(0.75, 1.0) if rng.random() < 0.2 else 1.0
            if qv and sx != (1 if qv > 0 else -1):
                shrink *= 0.8
            _m_ell(d, cx + sx * ex + qv, ey, ew * shrink, eh * shrink, lean, cy,
                   fill=(255, 255, 255, 255),
                   outline=ln, width=max(1, r // 18))
            d.ellipse([_m_shift(cx + sx * ex + qv + gaze - ew * 0.6 * shrink, ey - eh * 0.6,
                                0, cy, lean),
                       ey - eh * 0.6,
                       _m_shift(cx + sx * ex + qv + gaze + ew * 0.6 * shrink, ey + eh * 0.6,
                                0, cy, lean),
                       ey + eh * 0.6], fill=iris)
            if detail:
                _m_ell(d, cx + sx * ex + qv + gaze * 0.5, ey - eh * 0.35,
                       ew * 0.22, eh * 0.2, lean, cy, fill=(255, 255, 255, 255))

    # --- blush + mouth (skip blush at tiny sizes)
    if detail and rng.random() < 0.8:
        for sx in (-1, 1):
            _m_ell(d, cx + sx * ex + qv, ey + r * 0.42, r * 0.20, r * 0.11,
                   lean, cy, fill=(255, 150, 170, 160))
    mstyle = rng.choice(["smile", "open", "flat", "zig"])
    if mstyle == "smile":
        _m_arc(d, cx + qv, ey + r * 0.55, r * 0.25, r * 0.20, lean, cy,
               start=15, end=165, fill=ln, width=max(2, r // 14))
    elif mstyle == "open":
        _m_ell(d, cx + qv, ey + r * 0.52, r * 0.14, r * 0.14, lean, cy,
               fill=(140, 50, 50, 255), outline=ln)
    elif mstyle == "zig":
        _m_poly(d, [(cx - r * 0.15 + qv, ey + r * 0.5), (cx - r * 0.05 + qv, ey + r * 0.42),
                    (cx + r * 0.05 + qv, ey + r * 0.58), (cx + r * 0.15 + qv, ey + r * 0.5)],
                lean, cy, fill=None, outline=ln)
    else:
        _m_line(d, cx - r * 0.15 + qv, ey + r * 0.5, cx + r * 0.15 + qv, ey + r * 0.5,
                lean, cy, fill=ln, width=max(2, r // 14))

    if body:
        by0 = cy + hr * 0.95
        by1 = cy + hr * 0.95 + r * rng.uniform(0.55, 0.85)
        if rng.random() < 0.30:  # striped shirt variant
            _m_poly(d, [(cx - r * 0.55, by0), (cx + r * 0.55, by0),
                        (cx + r * 0.95, by1), (cx - r * 0.95, by1)],
                    lean, cy, fill=shirt, outline=ln)
            for k in range(rng.randint(1, 3)):
                yy = by0 + (by1 - by0) * (0.25 + 0.25 * k)
                _m_line(d, cx - r * 0.6, yy, cx + r * 0.6, yy, lean, cy,
                        fill=(245, 245, 245, 255) if sum(shirt[:3]) < 380 else (20, 20, 20, 255),
                        width=max(1, r // 18))
        else:
            _m_poly(d, [(cx - r * 0.55, by0), (cx + r * 0.55, by0),
                        (cx + r * 0.95, by1), (cx - r * 0.95, by1)],
                    lean, cy, fill=shirt, outline=ln)
        _m_poly(d, [(cx - r * 0.22, by0), (cx + r * 0.22, by0), (cx, by0 + r * 0.35)],
                lean, cy, fill=skin, outline=ln)
        if rng.random() < 0.15:  # scarf accessory
            scarf = rng.choice([(235, 40, 40), (255, 215, 0), (60, 120, 235)]) + (255,)
            _m_ell(d, cx, by0 + r * 0.1, r * 0.45, r * 0.16, lean, cy, fill=scarf)
    else:
        if rng.random() < 0.15:  # glasses on head-only busts
            for sx in (-1, 1):
                _m_ell(d, cx + sx * wr * 0.36 + qv, ey, r * 0.24, r * 0.24,
                       lean, cy, fill=None, outline=ln)


def _mascot_hood(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail):
    """Hooded cloak blob: arch silhouette + shadow face + glowing eyes. No hair/skin."""
    cloak = _m_desat(rng.choice(HOODS), rng) + (255,)
    trim = rng.choice(FURS) + (255,)
    glow = rng.choice(GLOWS) + (255,)
    bot = cy + (r * 1.9 if body else r * 1.1)

    # cloak mass: dome top + straight sides + jagged hem
    d.pieslice([_m_shift(cx - r * 1.05, cy - r * 1.5, 0, cy, lean), cy - r * 1.55,
                _m_shift(cx + r * 1.05, cy - r * 1.5, 0, cy, lean), cy + r * 0.6],
               start=180, end=360, fill=cloak, outline=ln)
    d.rectangle([_m_shift(cx - r * 1.05, cy - r * 0.2, 0, cy, lean), cy - r * 0.2,
                 _m_shift(cx + r * 1.05, cy - r * 0.2, 0, cy, lean), bot],
                fill=cloak, outline=ln)
    teeth = rng.randint(4, 6)
    hem = []
    for i in range(teeth * 2 + 1):
        x = cx - r * 1.05 + i * (r * 2.1 / (teeth * 2))
        y = bot if i % 2 == 0 else bot - r * rng.uniform(0.25, 0.45)
        hem.append((x, y))
    _m_poly(d, [(cx - r * 1.05, bot - r * 0.3)] + hem + [(cx + r * 1.05, bot - r * 0.3)],
            lean, cy, fill=cloak)
    _m_line(d, cx - r * 1.05, bot, cx + r * 1.05, bot, lean, cy, fill=ln, width=lw)

    # face opening (dark) + fur trim scallops
    _m_ell(d, cx, cy - r * 0.1, r * 0.68, r * 0.78, lean, cy,
           fill=(18, 18, 24, 255), outline=ln, width=max(1, lw - 1))
    if detail:
        for i in range(6):
            a = math.pi * (0.15 + 0.7 * i / 5)
            tx = cx + math.cos(a) * r * 0.68
            ty = cy - r * 0.1 - math.sin(a) * r * 0.78
            _m_ell(d, tx, ty, r * 0.14, r * 0.12, lean, cy, fill=trim)

    # glowing eyes: pair / slit / cyclops
    estyle = rng.choice(["pair", "pair", "slit", "cyclops"])
    ey = cy - r * 0.1
    if estyle == "cyclops":
        _m_ell(d, cx + gaze, ey, r * 0.20, r * 0.24, lean, cy, fill=glow)
        _m_ell(d, cx + gaze, ey, r * 0.09, r * 0.11, lean, cy, fill=(20, 20, 20, 255))
    elif estyle == "slit":
        _m_ell(d, cx - r * 0.28 + gaze, ey, r * 0.16, r * 0.07, lean, cy, fill=glow)
        _m_ell(d, cx + r * 0.28 + gaze, ey, r * 0.16, r * 0.07, lean, cy, fill=glow)
    else:
        er = r * rng.uniform(0.10, 0.15)
        _m_ell(d, cx - r * 0.28 + gaze, ey, er, er * 1.25, lean, cy, fill=glow)
        _m_ell(d, cx + r * 0.28 + gaze, ey, er, er * 1.25, lean, cy, fill=glow)
    if rng.random() < 0.6:
        _m_ell(d, cx + gaze, ey + r * 0.45, r * 0.09, r * 0.11, lean, cy,
               fill=(10, 10, 10, 255))


def _mascot_animal(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail):
    """Animal head: ears define silhouette, muzzle + nose instead of human mouth."""
    fur = _m_desat(rng.choice(FURS), rng) + (255,)
    cream = (245, 235, 220, 255)
    kind = rng.choice(["bear", "fox", "bunny", "bird"])
    hw, hh = r * 1.15, r * 0.9

    if kind == "bear":
        for sx in (-1, 1):
            _m_ell(d, cx + sx * r * 0.75, cy - r * 0.75, r * 0.38, r * 0.38,
                   lean, cy, fill=fur, outline=ln, width=lw)
            _m_ell(d, cx + sx * r * 0.75, cy - r * 0.75, r * 0.17, r * 0.17,
                   lean, cy, fill=cream)
    elif kind == "fox":
        for sx in (-1, 1):
            _m_poly(d, [(cx + sx * r * 0.4, cy - r * 0.6),
                        (cx + sx * r * 1.0, cy - r * 0.55),
                        (cx + sx * r * 0.55, cy - r * 1.35)],
                    lean, cy, fill=fur, outline=ln)
            _m_poly(d, [(cx + sx * r * 0.55, cy - r * 0.75),
                        (cx + sx * r * 0.8, cy - r * 0.72),
                        (cx + sx * r * 0.6, cy - r * 1.1)],
                    lean, cy, fill=cream)
    elif kind == "bunny":
        for sx in (-1, 1):
            tilt = sx * r * 0.15
            _m_ell(d, cx + sx * r * 0.4 + tilt, cy - r * 1.25, r * 0.26, r * 0.75,
                   lean, cy, fill=fur, outline=ln, width=lw)
            _m_ell(d, cx + sx * r * 0.4 + tilt, cy - r * 1.25, r * 0.12, r * 0.5,
                   lean, cy, fill=(250, 180, 190, 255))
    else:  # bird crest
        for i in range(3):
            sx = (i - 1) * r * 0.35
            _m_poly(d, [(sx + cx - r * 0.15, cy - hh * 0.9),
                        (sx + cx + r * 0.15, cy - hh * 0.9),
                        (sx + cx, cy - hh * 1.5 - r * 0.1 * (i % 2))],
                    lean, cy, fill=fur, outline=ln)

    # head + muzzle/beak
    _m_ell(d, cx, cy, hw, hh, lean, cy, fill=fur, outline=ln, width=lw)
    if kind == "bird":
        bc = rng.choice([(240, 150, 40), (235, 90, 60), (250, 200, 60)]) + (255,)
        _m_poly(d, [(cx - r * 0.3 + gaze, cy + r * 0.05),
                    (cx + r * 0.3 + gaze, cy + r * 0.05),
                    (cx + gaze, cy + r * 0.5)], lean, cy, fill=bc, outline=ln)
    else:
        _m_ell(d, cx + gaze * 0.5, cy + r * 0.42, r * 0.5, r * 0.32, lean, cy, fill=cream)
        nose = (30, 30, 30, 255)
        if rng.random() < 0.5:
            _m_poly(d, [(cx - r * 0.12 + gaze * 0.5, cy + r * 0.28),
                        (cx + r * 0.12 + gaze * 0.5, cy + r * 0.28),
                        (cx + gaze * 0.5, cy + r * 0.44)], lean, cy, fill=nose)
        else:
            _m_ell(d, cx + gaze * 0.5, cy + r * 0.34, r * 0.11, r * 0.08, lean, cy, fill=nose)
        _m_arc(d, cx + gaze * 0.5, cy + r * 0.42, r * 0.3, r * 0.18, lean, cy,
               start=15, end=165, fill=ln, width=max(1, lw - 1))

    # eyes: dots above muzzle + optional angry brows / whiskers
    er = r * rng.uniform(0.10, 0.16)
    for sx in (-1, 1):
        _m_ell(d, cx + sx * r * 0.45 + gaze, cy - r * 0.15, er, er * 1.2,
               lean, cy, fill=(25, 25, 25, 255))
        if detail and rng.random() < 0.7:
            _m_ell(d, cx + sx * r * 0.45 + gaze - er * 0.3, cy - r * 0.22,
                   er * 0.35, er * 0.35, lean, cy, fill=(255, 255, 255, 255))
    if rng.random() < 0.25:
        for sx in (-1, 1):
            _m_line(d, cx + sx * r * 0.2, cy - r * 0.45, cx + sx * r * 0.7, cy - r * 0.6,
                    lean, cy, fill=ln, width=max(1, lw - 1))
    if detail and kind in ("fox", "bear") and rng.random() < 0.5:
        for sx in (-1, 1):
            for k in range(2):
                _m_line(d, cx + sx * r * 0.8, cy + r * (0.3 + k * 0.15),
                        cx + sx * r * 1.3, cy + r * (0.25 + k * 0.15),
                        lean, cy, fill=ln, width=1)

    if body:
        by0 = cy + hh * 0.85
        _m_ell(d, cx, by0 + r * 0.45, r * 0.7, r * 0.5, lean, cy, fill=fur, outline=ln)


def _mascot_helmet(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail):
    """Angular helmet/bot: rect/dome/hex shell + single visor instead of two eyes."""
    helm = _m_desat(rng.choice(HELMS), rng) + (255,)
    accent = rng.choice(GLOWS) + (255,)
    dark = (18, 20, 26, 255)
    shape = rng.choice(["rect", "dome", "hex"])
    hw, hh = r * rng.uniform(0.85, 1.0), r * rng.uniform(0.85, 1.05)

    if shape == "rect":
        d.rounded_rectangle([_m_shift(cx - hw, cy - hh, 0, cy, lean), cy - hh,
                             _m_shift(cx + hw, cy - hh, 0, cy, lean), cy + hh],
                            radius=max(2, r // 8), fill=helm, outline=ln, width=lw)
    elif shape == "dome":
        d.pieslice([_m_shift(cx - hw, cy - hh * 1.2, 0, cy, lean), cy - hh * 1.25,
                    _m_shift(cx + hw, cy - hh * 1.2, 0, cy, lean), cy + hh * 0.4],
                   start=180, end=360, fill=helm, outline=ln)
        d.rectangle([_m_shift(cx - hw, cy - hh * 0.2, 0, cy, lean), cy - hh * 0.2,
                     _m_shift(cx + hw, cy - hh * 0.2, 0, cy, lean), cy + hh],
                    fill=helm, outline=ln)
    else:
        pts = []
        for i in range(6):
            a = math.pi / 6 + i * math.pi / 3
            pts.append((cx + hw * 1.1 * math.cos(a), cy + hh * 1.1 * math.sin(a)))
        _m_poly(d, pts, lean, cy, fill=helm, outline=ln)

    # antenna (often one-sided)
    if rng.random() < 0.5:
        sx = rng.choice([-1, 1]) if rng.random() < 0.5 else 0
        ax = cx + sx * hw * 0.5
        _m_line(d, ax, cy - hh, ax + sx * r * 0.2, cy - hh - r * 0.5,
                lean, cy, fill=ln, width=max(2, lw - 1))
        _m_ell(d, ax + sx * r * 0.2, cy - hh - r * 0.5, r * 0.10, r * 0.10,
               lean, cy, fill=accent, outline=ln)

    # visor slit + glow segments
    vh = r * rng.uniform(0.22, 0.32)
    vy = cy - r * rng.uniform(0.05, 0.25)
    vw = hw * rng.uniform(0.7, 0.85)
    d.rounded_rectangle([_m_shift(cx - vw + gaze, vy - vh, 0, cy, lean), vy - vh,
                         _m_shift(cx + vw + gaze, vy + vh, 0, cy, lean), vy + vh],
                        radius=int(vh), fill=dark, outline=ln, width=max(1, lw - 1))
    segs = rng.choice([1, 2, 2, 3])
    for i in range(segs):
        fx = -vw * 0.5 + i * (vw / max(1, segs - 1)) if segs > 1 else 0.0
        _m_ell(d, cx + fx + gaze, vy, r * 0.11, r * 0.11, lean, cy, fill=accent)

    # vents / mouth grille (no human mouth)
    for k in range(rng.randint(2, 3)):
        yy = cy + hh * 0.45 + k * r * 0.18
        _m_line(d, cx - r * 0.35, yy, cx + r * 0.35, yy, lean, cy,
                fill=ln, width=max(1, lw - 1))
    # side bolts
    for sx in (-1, 1):
        _m_ell(d, cx + sx * (hw + r * 0.02), cy, r * 0.14, r * 0.18,
               lean, cy, fill=accent, outline=ln)

    if body:
        by0 = cy + hh
        bh = r * rng.uniform(0.6, 0.9)
        d.rectangle([_m_shift(cx - hw * 0.8, by0, 0, cy, lean), by0,
                     _m_shift(cx + hw * 0.8, by0, 0, cy, lean), by0 + bh],
                    fill=helm, outline=ln)
        _m_ell(d, cx, by0 + bh * 0.4, r * 0.18, r * 0.18, lean, cy, fill=accent)


def _mascot_fullbody(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail):
    """Small head + big torso + stub limbs. Mass sits low, opposite of head-only chibi."""
    skin = _m_desat(rng.choice(SKINS), rng) + (255,)
    hair = _m_desat(rng.choice(HAIRS), rng) + (255,)
    shirt = _m_desat(rng.choice(SHIRTS), rng) + (255,)
    hr = r * rng.uniform(0.55, 0.75)
    hy = cy - r * 0.55
    cap = rng.choice(["arc", "beanie", "none"])

    if cap == "arc":
        d.pieslice([_m_shift(cx - hr * 1.05, hy - hr * 1.1, 0, cy, lean), hy - hr * 1.1,
                    _m_shift(cx + hr * 1.05, hy - hr * 1.1, 0, cy, lean), hy + hr * 0.3],
                   start=180, end=360, fill=hair, outline=ln)
    elif cap == "beanie":
        d.pieslice([_m_shift(cx - hr * 1.05, hy - hr * 1.2, 0, cy, lean), hy - hr * 1.2,
                    _m_shift(cx + hr * 1.05, hy - hr * 1.2, 0, cy, lean), hy + hr * 0.2],
                   start=180, end=360, fill=hair, outline=ln)
        d.rectangle([_m_shift(cx - hr * 1.05, hy - hr * 0.35, 0, cy, lean), hy - hr * 0.35,
                     _m_shift(cx + hr * 1.05, hy - hr * 0.35, 0, cy, lean), hy - hr * 0.05],
                    fill=hair, outline=ln)
    _m_ell(d, cx, hy, hr, hr, lean, cy, fill=skin, outline=ln, width=lw)
    # simple dot eyes (no white/highlight at this head size) + smile/flat
    er = max(1.5, hr * 0.13)
    for sx in (-1, 1):
        _m_ell(d, cx + sx * hr * 0.36 + gaze, hy, er, er * 1.2, lean, cy,
               fill=(25, 25, 25, 255))
    if rng.random() < 0.6:
        _m_arc(d, cx + gaze * 0.5, hy + hr * 0.25, hr * 0.3, hr * 0.22, lean, cy,
               start=15, end=165, fill=ln, width=max(1, lw - 1))
    else:
        _m_line(d, cx - hr * 0.25, hy + hr * 0.35, cx + hr * 0.25, hy + hr * 0.35,
                lean, cy, fill=ln, width=max(1, lw - 1))

    # torso + stub arms (+ legs when body)
    ty0 = hy + hr * 0.9
    th = r * rng.uniform(0.9, 1.4)
    tw = r * rng.uniform(0.6, 0.8)
    for sx in (-1, 1):
        _m_ell(d, cx + sx * (tw + r * 0.12), ty0 + th * 0.45, r * 0.16, r * 0.32,
               lean, cy, fill=shirt, outline=ln)
    d.rounded_rectangle([_m_shift(cx - tw, ty0, 0, cy, lean), ty0,
                         _m_shift(cx + tw, ty0, 0, cy, lean), ty0 + th],
                        radius=max(3, r // 6), fill=shirt, outline=ln, width=lw)
    if body:
        for sx in (-1, 1):
            lx = cx + sx * tw * 0.45
            d.rectangle([_m_shift(lx - r * 0.14, ty0 + th, 0, cy, lean), ty0 + th,
                         _m_shift(lx + r * 0.14, ty0 + th, 0, cy, lean),
                         ty0 + th + r * 0.35],
                        fill=(30, 30, 30, 255), outline=ln)


def _mascot_silhouette(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail):
    """Solid stamp blob: one fill, negative-space slit eyes, no thin-ring outline."""
    fill = _m_desat(rng.choice([(30, 30, 40), (170, 50, 60), (50, 110, 80),
                                (60, 90, 170), (200, 120, 50), (110, 70, 160)]), rng,
                    p=0.15) + (255,)
    tall = 1.8 if body else 1.15
    n = rng.randint(7, 10)
    pts = _blob_pts(cx, cy + r * 0.2, r * 1.0, r * tall * 0.62, n,
                    rng.uniform(0.08, 0.2), rng)
    if rng.random() < 0.35:
        # horns/spikes become part of the silhouette
        for sx in (-1, 1):
            pts.append((cx + sx * r * 0.5, cy - r * 1.1))
            pts.append((cx + sx * r * 0.3, cy - r * 0.6))
    _m_poly(d, pts, lean, cy, fill=fill)
    white = (245, 245, 245, 255)
    ey = cy + r * 0.05
    estyle = rng.choice(["slits", "dots", "happy"])
    if estyle == "happy":
        for sx in (-1, 1):
            _m_arc(d, cx + sx * r * 0.3 + gaze, ey, r * 0.2, r * 0.14,
                   lean, cy, start=180, end=360, fill=white, width=max(2, lw - 1))
    elif estyle == "dots":
        for sx in (-1, 1):
            _m_ell(d, cx + sx * r * 0.3 + gaze, ey, r * 0.11, r * 0.13,
                   lean, cy, fill=white)
    else:
        for sx in (-1, 1):
            d.pieslice([_m_shift(cx + sx * r * 0.3 + gaze - r * 0.16, ey - r * 0.06,
                                 0, cy, lean), ey - r * 0.06,
                        _m_shift(cx + sx * r * 0.3 + gaze + r * 0.16, ey + r * 0.06,
                                 0, cy, lean), ey + r * 0.06],
                       start=180, end=360, fill=white)
    if rng.random() < 0.5:
        _m_ell(d, cx + gaze * 0.5, cy + r * 0.75, r * 0.32, r * 0.24, lean, cy, fill=white)
    elif rng.random() < 0.5:
        _m_line(d, cx - r * 0.2 + gaze, ey + r * 0.4, cx + r * 0.2 + gaze, ey + r * 0.4,
                lean, cy, fill=white, width=max(2, lw - 1))


def _mascot_doodle(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail):
    """Wobbly hand-drawn figure: jittered outlines, mismatched eyes, scribble hair."""
    skin = _m_desat(rng.choice(SKINS), rng) + (255,)
    shirt = _m_desat(rng.choice(SHIRTS), rng) + (255,)
    thin = max(1, lw - 1)
    # head as jittered blob polygon (no clean ellipse)
    _m_poly(d, _blob_pts(cx, cy, r * rng.uniform(0.9, 1.05), r * rng.uniform(0.9, 1.05),
                         rng.randint(12, 16), 0.10, rng),
            lean, cy, fill=skin, outline=ln)
    # scribble hair: 3-5 unfilled arcs on top
    for _ in range(rng.randint(3, 5)):
        ox = rng.uniform(-r * 0.6, r * 0.6)
        _m_arc(d, cx + ox, cy - r * 0.75, r * rng.uniform(0.2, 0.4),
               r * rng.uniform(0.2, 0.35), lean, cy,
               start=rng.uniform(180, 220), end=rng.uniform(320, 360),
               fill=ln, width=thin)
    # mismatched eyes: one round, one slit/half-size
    ex = r * 0.36
    ey = cy + r * 0.08
    _m_ell(d, cx - ex + gaze, ey, r * 0.22, r * 0.28, lean, cy,
           fill=(255, 255, 255, 255), outline=ln, width=thin)
    _m_ell(d, cx - ex + gaze, ey, r * 0.10, r * 0.13, lean, cy, fill=(25, 25, 25, 255))
    if rng.random() < 0.5:
        _m_ell(d, cx + ex + gaze, ey, r * 0.16, r * 0.08, lean, cy, fill=ln)
    else:
        _m_ell(d, cx + ex + gaze, ey, r * 0.14, r * 0.18, lean, cy,
               fill=(25, 25, 25, 255))
    # scribble mouth
    _m_poly(d, [(cx - r * 0.18 + gaze, ey + r * 0.5),
                (cx - r * 0.06 + gaze, ey + r * (0.42 if rng.random() < 0.5 else 0.58)),
                (cx + r * 0.06 + gaze, ey + r * 0.5),
                (cx + r * 0.18 + gaze, ey + r * (0.44 if rng.random() < 0.5 else 0.56))],
            lean, cy, fill=None, outline=ln)
    if body:
        by0 = cy + r * 0.95
        by1 = by0 + r * rng.uniform(0.6, 0.85)
        _m_poly(d, _blob_pts(cx, (by0 + by1) / 2, r * 0.8, (by1 - by0) / 2, 10, 0.12, rng),
                lean, cy, fill=shirt, outline=ln)


def draw_mascot(d: ImageDraw.ImageDraw, cx, cy, r, rng: random.Random,
                body: bool = True):
    """Multi-archetype mascot dispatcher (anti-overfit rewrite).

    Picks one of 7 structurally distinct families — chibi / hood-blob /
    animal / helmet-bot / fullbody / silhouette-stamp / doodle — each with its
    own silhouette, eye topology (pair / visor / slit / cyclops / dots),
    stroke weight/color, and pose lean. All geometric, all from the caller's
    seeded rng. r = head radius; total height ~2*r..4*r (body=False: head only).
    """
    r = max(8, int(r * rng.uniform(0.85, 1.20))) if rng.random() < 0.3 else max(10, int(r))
    ln, lw = _m_stroke(rng, r)
    lean = rng.uniform(-0.35, 0.35) if rng.random() < 0.75 else 0.0
    gaze = rng.uniform(-0.18, 0.18) * r if rng.random() < 0.65 else 0.0
    detail = r >= 16
    x = rng.random()
    if x < 0.18:
        _mascot_chibi(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail)
    elif x < 0.32:
        _mascot_hood(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail)
    elif x < 0.46:
        _mascot_animal(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail)
    elif x < 0.60:
        _mascot_helmet(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail)
    elif x < 0.74:
        _mascot_fullbody(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail)
    elif x < 0.87:
        _mascot_silhouette(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail)
    else:
        _mascot_doodle(d, cx, cy, r, rng, body, ln, lw, lean, gaze, detail)


def _badge_body_fill(rng: random.Random):
    """Diversified badge body color. Breaks the always-grey signature."""
    roll = rng.random()
    if roll < 0.32:
        rgb = (15, 15, 15)
    elif roll < 0.52:
        rgb = (245, 245, 245)
    else:
        rgb = rng.choice([(200, 30, 30), (30, 90, 200), (150, 60, 160),
                          (255, 122, 0), (30, 130, 70), (40, 40, 55)])
    return rgb, rng.randint(120, 200)


def _centered_line(d, text, font, top_y, W, fill, sc, sw, rng, jitter=8):
    """Draw one centered text line at top_y. Returns its pixel height."""
    bb = font.getbbox(text)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    x = (W - tw) // 2 - bb[0] + rng.randint(-jitter, jitter)
    x = max(4 - bb[0], min(x, W - 4 - tw - bb[0]))
    if sw:
        d.text((x, top_y - bb[1]), text, font=font, fill=fill + (255,),
               stroke_width=sw, stroke_fill=_flat_stroke(sc) + (255,))
    else:
        d.text((x, top_y - bb[1]), text, font=font, fill=fill + (255,))
    return th


def gen_number_badge(rng: random.Random, np_rng) -> Image.Image:
    num = rng.randint(1, 9999) if rng.random() < 0.25 else rng.randint(1, 999)
    brand = fake_brand(rng)
    W = rng.randint(120, 320)
    H = rng.randint(120, 320)
    shape_roll = rng.random()
    if shape_roll < 0.25:
        H = max(H, int(W * 1.3))  # tall badge
    elif shape_roll < 0.50:
        W = max(W, int(H * 1.3))  # wide badge
    elif shape_roll < 0.58:
        H = max(H, int(W * 1.6))  # extra-tall seal
    elif shape_roll < 0.64:
        W = max(W, int(H * 1.6))  # extra-wide banner-badge
    area = W * H
    large = area > 35000
    im, d = new_canvas(W, H)
    # --- badge body: silhouette x fill treatment, never always grey round rect
    body_rgb, body_alpha = _badge_body_fill(rng)
    body = body_rgb + (body_alpha,)
    lum = 0.299 * body_rgb[0] + 0.587 * body_rgb[1] + 0.114 * body_rgb[2]
    dark = lum < 150
    edge = (255, 255, 255, 220) if dark else (10, 10, 10, 220)
    if rng.random() < 0.25:  # colored edge instead of white/black
        edge = rng.choice([(235, 40, 40), (255, 215, 0), (40, 160, 255),
                           (255, 255, 255), (10, 10, 10)]) + (220,)
    ew = rng.choices([1, 2, 3, 4, 5], weights=[15, 30, 30, 15, 10])[0]
    bstyle = rng.choices(
        ["round", "shield", "disc", "double", "torn", "outline", "panel",
         "sharp", "chamfer", "taper", "slant", "hex", "octagon", "capsule",
         "asym_round", "wavy", "scallop", "seal", "ticket", "bookmark", "blob2",
         "hexcut", "concentric", "fold", "tornedge"],
        weights=[7, 5, 5, 5, 5, 6, 5,
                 4, 4, 5, 4, 4, 4, 4,
                 5, 5, 4, 4, 4, 4, 4,
                 4, 4, 4, 4])[0]
    hollow = bstyle in ("outline",) or (bstyle in ("wavy", "chamfer", "sharp") and rng.random() < 0.18)
    if bstyle == "round":
        d.rounded_rectangle([4, 4, W - 4, H - 4], radius=rng.randint(10, 30), fill=body,
                            outline=edge, width=ew + 1)
    elif bstyle == "double":
        d.rounded_rectangle([4, 4, W - 4, H - 4], radius=rng.randint(10, 30), fill=body,
                            outline=edge, width=2)
        d.rounded_rectangle([10, 10, W - 10, H - 10], radius=rng.randint(6, 20), fill=None,
                            outline=edge, width=1)
    elif bstyle == "disc":
        d.ellipse([6, 6, W - 6, H - 6], fill=body, outline=edge, width=ew + 1)
    elif bstyle == "shield":
        d.polygon([(10, 6), (W - 10, 6), (W - 6, H * 0.55), (W / 2, H - 6), (6, H * 0.55)],
                  fill=body, outline=edge)
    elif bstyle == "outline":
        # transparent interior: no flat plateau, text + graphics carry it
        d.rounded_rectangle([4, 4, W - 4, H - 4], radius=rng.randint(10, 30), fill=None,
                            outline=edge, width=rng.randint(3, 5))
        dark = False  # interior transparent -> draw text as on-page, contrast via stroke
    elif bstyle == "panel":
        d.rounded_rectangle([4, 4, W - 4, H - 4], radius=rng.randint(10, 30), fill=body,
                            outline=edge, width=ew + 1)
        panel_rgb = (245, 245, 245) if dark else (15, 15, 15)
        if rng.random() < 0.3:
            panel_rgb = rng.choice([(200, 30, 30), (30, 90, 200), (255, 215, 0)])
        d.rounded_rectangle([12, 12, W - 12, H - 30], radius=rng.randint(6, 14),
                            fill=panel_rgb + (rng.randint(90, 160),))
        dark = (panel_rgb[0] * 0.299 + panel_rgb[1] * 0.587 + panel_rgb[2] * 0.114) < 150
    elif bstyle == "torn":
        d.polygon(_blob_pts(W / 2, H / 2, W / 2 - 6, H / 2 - 6, 12, 0.06, rng),
                  fill=body, outline=edge)
    else:
        # --- new varied silhouette family (all polygon-based, shared helpers)
        if bstyle == "sharp":
            pts = [(5, 5), (W - 5, 5), (W - 5, H - 5), (5, H - 5)]
        elif bstyle == "chamfer":
            pts = _chamfer_rect_pts(W, H, rng)
        elif bstyle == "taper":
            pts = _taper_pts(W, H, rng)
        elif bstyle == "slant":
            pts = _slant_pts(W, H, rng)
        elif bstyle == "hex":
            pts = _blob_pts(W / 2, H / 2, W / 2 - 6, H / 2 - 6, 6, 0.02, rng)
        elif bstyle == "octagon":
            pts = _blob_pts(W / 2, H / 2, W / 2 - 6, H / 2 - 6, 8, 0.02, rng)
        elif bstyle == "capsule":
            # stadium/pill: uniform large radius, orientation follows longer axis
            d.rounded_rectangle([4, 4, W - 4, H - 4],
                                radius=min(W, H) // 2 - 4, fill=None if hollow else body,
                                outline=edge, width=(rng.randint(3, 5) if hollow else ew + 1))
            pts = []
        elif bstyle == "asym_round":
            pts = _asym_round_pts(W, H, rng)
        elif bstyle == "wavy":
            pts = _wavy_rect_pts(W, H, rng)
        elif bstyle == "scallop":
            pts = _scallop_pts(W / 2, H / 2, W / 2 - 6, H / 2 - 6, rng)
        elif bstyle == "seal":
            pts = _seal_pts(W / 2, H / 2, W / 2 - 6, H / 2 - 6, rng)
        elif bstyle == "ticket":
            pts = _ticket_pts(W, H, rng)
        elif bstyle == "bookmark":
            pts = _bookmark_pts(W, H, rng)
        elif bstyle == "hexcut":  # hexagon with 2 cut corners
            raw = _blob_pts(W / 2, H / 2, W / 2 - 6, H / 2 - 6, 6, 0.02, rng)
            pts = raw[:4] + [(W - 5, H - 5), (5, H - 5)]
        elif bstyle == "concentric":
            d.ellipse([4, 4, W - 4, H - 4], fill=body, outline=edge, width=ew + 1)
            d.ellipse([14, 14, W - 14, H - 14], fill=None, outline=edge, width=1)
            d.ellipse([22, 22, W - 22, H - 22], fill=None, outline=edge, width=1)
            pts = []
        elif bstyle == "fold":  # folded corner (dog-ear)
            fold = min(30, min(W, H) * 0.25)
            pts = [(5, 5), (W - 5 - fold, 5), (W - 5, 5 + fold),
                   (W - 5, H - 5), (5, H - 5)]
        elif bstyle == "tornedge":  # heavy-jitter torn edge
            pts = _blob_pts(W / 2, H / 2, W / 2 - 6, H / 2 - 6, 14, 0.14, rng)
        else:  # blob2: loose random vector path, more organic than torn
            pts = _random_path_pts(W / 2, H / 2, W / 2 - 6, H / 2 - 6, rng)
        if pts:
            if rng.random() < 0.40:  # lean modifier, breaks mirror symmetry
                pts = _shear_pts(pts, rng)
            if rng.random() < 0.30:  # hand-drawn wobble modifier
                pts = _roughen_pts(pts, rng, amt=rng.uniform(1.5, 3.5))
            if hollow:
                d.polygon(pts, fill=None, outline=edge, width=rng.randint(3, 5))
                dark = False
            else:
                _stroke_poly(d, pts, body, edge)
    if bstyle not in ("outline", "ticket", "bookmark", "seal", "scallop",
                        "concentric", "fold") \
            and not hollow and rng.random() < 0.25:
        # split-body: bottom half second tone + divider, breaks flat void
        c2 = rng.choice([(200, 30, 30), (30, 90, 200), (15, 15, 15),
                         (245, 245, 245), (255, 122, 0)])
        mid = int(H * rng.uniform(0.45, 0.65))
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([6, mid, W - 6, H - 6], fill=c2 + (rng.randint(80, 150),))
        im.alpha_composite(overlay)
        d = ImageDraw.Draw(im)
        d.line([(8, mid), (W - 8, mid)], fill=edge, width=1)
    # graphic zone: bigger marks, rarely empty on top
    horizontal = W > H * 1.25
    gfx = rng.choices(["mascot", "logo", "both", "none"], weights=[40, 38, 18, 4])[0]
    gfx_bottom = 0
    text_x = 12
    if horizontal:
        gcx = int(W * rng.uniform(0.12, 0.25))
        gcy = H // 2 + rng.randint(-H // 10, H // 10)
        gr = max(10, int(H * rng.uniform(0.20, 0.30)))
        text_x = min(W - 60, gcx + gr + rng.randint(8, 18))
    else:
        corner = rng.random()
        if corner < 0.50:
            gcx = W // 2 + rng.randint(-W // 6, W // 6)
        elif corner < 0.75:
            gcx = int(W * rng.uniform(0.22, 0.32))
        else:
            gcx = int(W * rng.uniform(0.68, 0.78))
        gr = max(11, int(min(W, H) * rng.uniform(0.12, 0.20)))
        gcy = int(8 + gr * 1.1)
    if gfx == "mascot":
        draw_mascot(d, gcx, gcy, gr, rng, body=rng.random() < 0.7)
        gfx_bottom = gcy + int(gr * 2.0)
    elif gfx == "logo":
        draw_logo(im, gcx, gcy, gr, rng)
        d = ImageDraw.Draw(im)
        gfx_bottom = gcy + gr + 4
    elif gfx == "both":
        mr = max(8, int(gr * 0.8))
        draw_mascot(d, gcx - gr, gcy, mr, rng, body=rng.random() < 0.5)
        draw_logo(im, gcx + gr, gcy, mr, rng)
        d = ImageDraw.Draw(im)
        gfx_bottom = gcy + gr + 4
    ink = (255, 255, 255) if dark else (10, 10, 10)
    # reserve bottom URL strip so the flow never collides with it.
    # horizontal gfx is sided, not stacked -> text column flows from the top.
    url_h, url_y0 = 20, H - 32
    y = 10 if horizontal else (max(8, gfx_bottom + 4) if gfx_bottom else 10)
    # --- tagline(s): area-proportional, fills the old void above the number
    n_tag = 2 if (large and rng.random() < 0.7) else 1
    if H > 170 and n_tag < 2 and rng.random() < 0.5:
        n_tag = 2
    if H > 240 and rng.random() < 0.5:
        n_tag += 1
    tag_pool = KR_LINES + CN_LINES + BADGE_TAGLINES if rng.random() < 0.5 else BADGE_TAGLINES
    for _ in range(n_tag):
        if y > url_y0 - 70:
            break
        cr = rng.random()
        if cr < 0.15:
            t = _compose_cjk(rng, rng.choice(["kr", "cn", "jp"]))
        elif cr < 0.25:
            t = _decorate(rng.choice(tag_pool), rng, p=1.0)
        elif cr < 0.32:
            t = _mixed_line(rng)
        else:
            t = rng.choice(tag_pool)
        f = fit_font_size(t, (W - text_x - 12) if horizontal else W - 24,
                          rng.randint(15, 24))
        bb = f.getbbox(t)
        fill = ink if rng.random() < 0.6 else pick_text_color(rng)
        sc, sw = outline_for(fill, rng)
        if hollow or (fill == ink and rng.random() < 0.5):
            sw = 0 if rng.random() < 0.5 else 1
        tx = (text_x - bb[0]) if horizontal else (W - (bb[2] - bb[0])) // 2 - bb[0]
        draw_fancy_text(im, (tx, y - bb[1]), t, f, fill, sc if sw else None, sw, rng=rng)
        d = ImageDraw.Draw(im)
        y += (bb[3] - bb[1]) + 3
    # --- number label: 4 formats, jittered size/position, sometimes strokeless
    fmt = rng.random()
    if fmt < 0.42:
        label = f"{brand[:3].upper()}{num}"
    elif fmt < 0.65:
        label = f"{num}"
    elif fmt < 0.85:
        label = f"{brand.upper()} {num}"
    else:
        label = f"NO.{num} {_decorate(brand.upper(), rng, p=1.0)}"
    zone_w = (W - text_x - 12) if horizontal else (W - 24)
    fnum = fit_font_size(label, max(40, zone_w), rng.randint(36, 62))
    bb = fnum.getbbox(label)
    fill = pick_text_color(rng)
    sc, sw = outline_for(fill, rng)
    if rng.random() < 0.25:  # flat print on contrasting body, no stroke
        sw = 0
    if horizontal:
        num_x = text_x - bb[0]
        num_y = min(max(y, int(H * 0.25)), url_y0 - (bb[3] - bb[1]) - 40) - bb[1]
        y = num_y + bb[1] + (bb[3] - bb[1]) + 3
    else:
        num_x = (W - (bb[2] - bb[0])) // 2 - bb[0] + rng.randint(-W // 10, W // 10)
        num_y = max(y, int(H * 0.30)) if not gfx_bottom else max(y, gfx_bottom + 2)
        num_y = min(num_y, url_y0 - (bb[3] - bb[1]) - 36) - bb[1]
    draw_fancy_text(im, (num_x, num_y), label, fnum, fill, sc if sw else None, sw,
                    rng=rng, grad_p=0.30, shadow_p=0.40)
    d = ImageDraw.Draw(im)
    # flank narrow numbers: side rules + glyph dots fill horizontal void
    num_w = bb[2] - bb[0]
    if not horizontal and num_w < zone_w * 0.55 and num_y > 12:
        gy = num_y + bb[1] + (bb[3] - bb[1]) // 2
        gx0 = max(10, int(num_x + bb[0] - 14 - zone_w * 0.18))
        gx1 = min(W - 10, int(num_x + bb[0] + num_w + 14 + zone_w * 0.18))
        gl = rng.choice(["★", "◆", "●", "✦"])
        gf = load_font_for(gl, max(10, (bb[3] - bb[1]) // 2))
        gb = gf.getbbox(gl)
        for sx, ex in ((gx0, int(num_x + bb[0] - 10)), (int(num_x + bb[0] + num_w + 10), gx1)):
            if ex - sx > 24:
                d.line([(sx, gy), (ex, gy)], fill=edge, width=2)
                d.text((sx - (gb[2] - gb[0]) // 2 - gb[0], gy - (gb[3] - gb[1]) // 2 - gb[1]),
                       gl, font=gf, fill=edge)
    if not horizontal:
        y = num_y + bb[1] + (bb[3] - bb[1]) + 3
    if fmt >= 0.42 and fmt < 0.65:  # big bare number -> small brand line above/below
        sub = brand.upper()
        fs = load_font_for(sub, rng.randint(11, 15))
        if bstyle in ("disc", "seal", "scallop", "round") and rng.random() < 0.25:
            # arc seal text on round badges (uses the previously dead arc path)
            arc = _render_arc_tile(sub, fs, ink, None, 0, rng)
            ax = (W - arc.width) // 2
            if y + arc.height < url_y0 - 20:
                _comp_tile_late(im, arc, ax, int(y))
                d = ImageDraw.Draw(im)
                y += arc.height + 3
        else:
            sb = fs.getbbox(sub)
            sx = text_x - sb[0] if horizontal else (W - (sb[2] - sb[0])) // 2 - sb[0]
            sy = y - sb[1]
            if sy < url_y0 - 30:
                d.text((sx, sy), sub, font=fs, fill=ink + (255,))
                y += (sb[3] - sb[1]) + 3
    if rng.random() < 0.65 and y < url_y0 - 34:
        # divider rule: breaks the flat interior into text bands
        d.line([(12, y), (W - 12, y)], fill=edge, width=1)
        y += 5
    # --- info line(s): domain variants / update lines fill mid-body
    n_info = 2 if (large or H > 170) else 1
    if large and rng.random() < 0.4:
        n_info += 1
    for _ in range(n_info):
        if y > url_y0 - 34:
            break
        pick = rng.random()
        if pick < 0.35:
            t = f"{brand}{rng.randint(1, 99)}{rng.choice(TLDS)}".upper()
        elif pick < 0.55:
            t = rng.choice(KR_LINES + CN_LINES + ROMAN_LINES)
        elif pick < 0.65:
            t = _compose_cjk(rng, rng.choice(["kr", "cn", "jp"]))
        elif pick < 0.75:
            t = _mixed_line(rng)
        else:
            t = rng.choice(BADGE_TAGLINES)
        f = fit_font_size(t, (W - text_x - 12) if horizontal else W - 28, rng.randint(12, 19))
        bb3 = f.getbbox(t)
        tx = (text_x - bb3[0]) if horizontal else (W - (bb3[2] - bb3[0])) // 2 - bb3[0]
        d.text((tx, y - bb3[1]), t, font=f, fill=ink + (255,))
        y += (bb3[3] - bb3[1]) + 3
    # roll URL placement FIRST so the fill budget below knows the bottom limit.
    # small url: bottom (usual), top, or omitted; TLD varies
    url_roll = rng.random()
    if url_roll < 0.70:
        tld = ".com" if rng.random() < 0.6 else rng.choice(TLDS)
        pre = "WWW." if rng.random() < 0.2 else ""
        url = f"{pre}{brand}{num}{tld}"
        fsmall = load_font_for(url, rng.randint(12, 17))
        url_pos = "bottom"
    elif url_roll < 0.85:
        url, url_pos = "", "none"  # no url: number-only badge
    else:
        url = f"{brand}{rng.choice(TLDS)}".upper()
        fsmall = load_font_for(url, rng.randint(10, 13))
        url_pos = "top"
    bottom_lim = (H - 8) if url_pos == "none" else (url_y0 - 10)
    # --- smallprint paragraph: 3-4 micro lines on large badges
    if large or H > 190:
        n_sp = 3 if (area > 45000 and rng.random() < 0.7) else 2
        if (large or H > 210) and rng.random() < 0.4:
            n_sp += 1
        if horizontal:
            n_sp = min(n_sp, 1)
        for _ in range(n_sp):
            if y > bottom_lim:
                break
            t = rng.choice(BADGE_SMALLPRINT)
            f = load_font_for(t, rng.randint(10, 14))
            bb4 = f.getbbox(t)
            if (bb4[2] - bb4[0]) > (W - 28):
                f = fit_font_size(t, W - 28, 14)
                bb4 = f.getbbox(t)
            tx = (W - (bb4[2] - bb4[0])) // 2 - bb4[0] + rng.randint(-6, 6)
            d.text((tx, y - bb4[1]), t, font=f, fill=ink + (255,))
            y += (bb4[3] - bb4[1]) + 2
    # --- vertical fill pass: eat leftover whitespace with extra lines so the
    # body never sits half-empty (real badges pack text edge to edge).
    # Runs until the text column reaches the bottom limit, alternating
    # divider rules and micro lines.
    for _ in range(5):
        if y > bottom_lim - 8:
            break
        if rng.random() < 0.30 and y < bottom_lim - 26:
            d.line([(12, y), (W - 12, y)], fill=edge, width=1)
            y += 5
            continue
        t = rng.choice(BADGE_SMALLPRINT + BADGE_TAGLINES)
        f = fit_font_size(t, W - 28, 14)
        bb5 = f.getbbox(t)
        if (bb5[3] - bb5[1]) <= 0:
            break
        tx = (W - (bb5[2] - bb5[0])) // 2 - bb5[0] + rng.randint(-6, 6)
        fill5 = ink if rng.random() < 0.7 else pick_text_color(rng)
        d.text((tx, y - bb5[1]), t, font=f, fill=fill5 + (255,))
        y += (bb5[3] - bb5[1]) + 2
    if url_pos == "bottom":
        bb2 = fsmall.getbbox(url)
        ux = (W - (bb2[2] - bb2[0])) // 2 - bb2[0] + rng.randint(-8, 8)
        d.text((ux, H - (bb2[3] - bb2[1]) - 12 - bb2[1]),
               url, font=fsmall, fill=ink + (255,))
    elif url_pos == "top":
        bb2 = fsmall.getbbox(url)
        d.text(((W - (bb2[2] - bb2[0])) // 2 - bb2[0], 8 - bb2[1]),
               url, font=fsmall, fill=ink + (255,))
    return im


def gen_ribbon_bar(rng: random.Random, np_rng) -> Image.Image:
    W = rng.randint(320, 900)
    H = rng.randint(40, 140)
    two_line = H >= 75 and rng.random() < 0.65
    tag = _decorate(rng.choice(ROMAN_LINES + BADGE_TAGLINES), rng, p=0.20)
    dom = fake_domain(rng).upper()
    im, d = new_canvas(W, H)
    # 15% broken-bar: no flat plateau, just outlined text + rules
    if rng.random() < 0.15:
        f = fit_font_size(f"{tag} {dom}", W - 30, H - 22)
        bb = f.getbbox(f"{tag} {dom}")
        fill = pick_text_color(rng)
        sc, sw = outline_for(fill, rng)
        draw_fancy_text(im, ((W - (bb[2] - bb[0])) // 2 - bb[0],
                             (H - (bb[3] - bb[1])) // 2 - bb[1]),
                         f"{tag} {dom}", f, fill, sc, sw, rng=rng)
        y = (H - (bb[3] - bb[1])) // 2 - 8
        d.line([(20, y), (W - 20, y)], fill=(255, 255, 255, 180), width=1)
        d.line([(20, y + (bb[3] - bb[1]) + 12), (W - 20, y + (bb[3] - bb[1]) + 12)],
               fill=(255, 255, 255, 180), width=1)
        return im
    bar_color = rng.choice([(10, 10, 10), (240, 240, 240), (200, 30, 30), (30, 90, 200),
                            (150, 60, 160), (30, 130, 70), (255, 122, 0),
                            (40, 40, 55)])
    bar_color = _hsl_jitter(bar_color, rng) if rng.random() < 0.5 else bar_color
    bar_alpha = rng.randint(50, 210)
    # bar silhouette: rect / notch / split / pill / taper
    bform = rng.choices(["rect", "notch", "split", "pill", "taper"],
                        weights=[45, 15, 15, 15, 10])[0]
    rule_x1 = W  # top/bottom rules stop where the bar's straight edges stop
    if bform == "rect":
        d.rectangle([0, 6, W, H - 6], fill=bar_color + (bar_alpha,))
    elif bform == "notch":
        pt = min(W // 8 + 8, H)
        mid = H // 2
        d.polygon([(0, 6), (W - pt, 6), (W, mid), (W - pt, H - 6), (0, H - 6)],
                  fill=bar_color + (bar_alpha,))
        rule_x1 = W - pt
    elif bform == "pill":
        rad = (H - 12) // 2
        d.rounded_rectangle([0, 6, W, H - 6], radius=max(2, rad),
                            fill=bar_color + (bar_alpha,))
        rule_x1 = W - rad
    elif bform == "taper":
        inset = min(W * 0.05, (H - 12) * 0.3)
        d.polygon([(0, 6), (W, 6), (W - inset, H - 6), (0 + inset, H - 6)],
                  fill=bar_color + (bar_alpha,))
    else:
        mid = H // 2
        d.rectangle([0, 6, W, mid - 2], fill=bar_color + (bar_alpha,))
        d.rectangle([0, mid + 2, W, H - 6], fill=bar_color + (max(40, bar_alpha - 40),))
    d.line([(0, 6), (rule_x1, 6)], fill=(255, 255, 255, 200), width=2)
    d.line([(0, H - 6), (rule_x1, H - 6)], fill=(255, 255, 255, 200), width=2)
    # side logo fills the old empty bar ends on wide bars
    side = rng.random() < 0.55 and W > 440
    tx0, avail = 15, W - 30
    if side:
        lr = max(10, int((H - 16) * 0.42))
        lx = lr + 12 if rng.random() < 0.7 else W - lr - 12
        draw_logo(im, lx, H // 2, lr, rng)
        d = ImageDraw.Draw(im)
        if lx < W // 2:
            tx0, avail = lx + lr + 10, W - (lx + lr + 10) - 15
        else:
            tx0, avail = 15, lx - lr - 10 - 15
    fill = (255, 255, 255) if sum(bar_color) < 300 else (15, 15, 15)
    if two_line:
        ftag = fit_font_size(tag, avail, max(11, H // 4))
        fdom = fit_font_size(dom, avail, max(12, H // 3 + 6))
        tb = ftag.getbbox(tag)
        db = fdom.getbbox(dom)
        total = (tb[3] - tb[1]) + (db[3] - db[1]) + 6
        y = (H - total) // 2
        draw_fancy_text(im, (tx0 + max(0, (avail - (tb[2] - tb[0])) // 2) - tb[0], y - tb[1]),
                        tag, ftag, fill, None, 0, rng=rng)
        d = ImageDraw.Draw(im)
        y += (tb[3] - tb[1]) + 6
        draw_fancy_text(im, (tx0 + max(0, (avail - (db[2] - db[0])) // 2) - db[0], y - db[1]),
                        dom, fdom, fill, None, 0, rng=rng)
        d = ImageDraw.Draw(im)
    else:
        text = f"{tag} {dom}"
        f = fit_font_size(text, avail, H - 22)
        bb = f.getbbox(text)
        draw_fancy_text(im, (tx0 + max(0, (avail - (bb[2] - bb[0])) // 2) - bb[0],
                             (H - (bb[3] - bb[1])) // 2 - bb[1]),
                        text, f, fill, None, 0, rng=rng)
        d = ImageDraw.Draw(im)
    return im


# ---------------------------------------------------------------- randomized logo mark
# Anti-overfit rewrite (v2): real WM logos vary in palette (mono / pastel /
# dark, not just vivid), geometry (stretched / tilted / sheared / off-center,
# not just concentric), fill (nested-gradient / hollow / striped / duo echo,
# not just flat) and composition (backing container + mark + micro detail).
# Tile-based: each logo renders onto a small local RGBA tile (so text can tilt
# and shadows/glows can layer) which is then composited onto the host image.
# Signature is draw_logo(host_im, cx, cy, r, rng) — host must be RGBA.

_LOGO_VIVID = [(235, 40, 40), (255, 122, 0), (255, 200, 0), (60, 200, 120),
               (50, 140, 255), (150, 90, 255), (255, 90, 180), (240, 240, 240),
               (0, 180, 180), (140, 200, 40), (200, 60, 140), (90, 90, 200),
               (255, 140, 60), (40, 200, 140), (120, 80, 200), (230, 230, 230)]
_LOGO_PASTEL = [(255, 170, 170), (255, 200, 140), (255, 235, 150), (170, 230, 180),
                (150, 200, 255), (200, 170, 255), (255, 180, 220), (220, 220, 220)]
_LOGO_DARK = [(140, 30, 30), (160, 80, 20), (150, 120, 20), (30, 120, 70),
              (30, 80, 160), (90, 60, 150), (160, 50, 110), (60, 60, 70)]
_LOGO_MONO = [(15, 15, 15), (45, 45, 45), (120, 120, 120),
              (200, 200, 200), (245, 245, 245)]
_LOGO_GLYPHS = "ABCDEFGHJKLMNPRSTVWXYZ23456789"


def _shade(rgb, f):
    return (max(0, min(255, int(rgb[0] * f))), max(0, min(255, int(rgb[1] * f))),
            max(0, min(255, int(rgb[2] * f))))


def _lerp3(a, b, t):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def _logo_palette(rng: random.Random) -> dict:
    """Two main colors + backing + outline treatment. Breaks the old
    always-vivid + always-black-stroke signature."""
    m = rng.random()
    if m < 0.30:
        c1, c2 = rng.choice(_LOGO_VIVID), rng.choice(_LOGO_VIVID)
    elif m < 0.50:
        c1, c2 = rng.choice(_LOGO_MONO), rng.choice(_LOGO_MONO)
        if rng.random() < 0.45:  # mono + one vivid accent reads very "real logo"
            c2 = rng.choice(_LOGO_VIVID[:7])
    elif m < 0.65:
        c1, c2 = rng.choice(_LOGO_PASTEL), rng.choice(_LOGO_PASTEL)
    elif m < 0.80:
        c1, c2 = rng.choice(_LOGO_DARK), rng.choice(_LOGO_DARK)
    else:
        c1 = rng.choice(_LOGO_VIVID)
        c2 = _shade(c1, rng.uniform(0.45, 0.65)) if rng.random() < 0.5 \
            else _lerp3(c1, (255, 255, 255), rng.uniform(0.35, 0.6))
    if rng.random() < 0.25:  # single-hue marks are common in real WMs
        c2 = _shade(c1, rng.uniform(0.6, 0.85)) if rng.random() < 0.5 \
            else _lerp3(c1, (255, 255, 255), 0.45)
    back = rng.choice(_LOGO_DARK + [(12, 12, 12), (240, 240, 240)])
    if rng.random() < 0.3:
        back = _shade(c1, 0.45)
    om = rng.random()
    base_lw = rng.choice([1, 2, 2, 3])
    if om < 0.38:
        oc, ow, double = (12, 12, 12), base_lw + 1, False
    elif om < 0.58:
        oc, ow, double = (255, 255, 255), base_lw + 1, False
    elif om < 0.73:
        oc, ow, double = _shade(c1, 0.5), base_lw, False
    elif om < 0.85:
        oc, ow, double = ((255, 255, 255) if sum(c1) < 380 else (12, 12, 12)), base_lw, True
    else:
        oc, ow, double = None, 0, False  # no stroke: fill/shadow/glow must carry it
    shadow = (rng.randint(1, 3), rng.randint(1, 3)) if rng.random() < 0.30 else None
    glow = rng.choice(_LOGO_VIVID) + (70,) if rng.random() < 0.12 else None
    return {"c1": c1, "c2": c2, "back": back, "oc": oc, "ow": ow,
            "double": double, "shadow": shadow, "glow": glow}


def _logo_xform(rng: random.Random) -> dict:
    """Per-logo geometric params: independent x/y stretch, rotation, shear,
    mirror, off-center shift. Applied to every point before drawing."""
    return {"sx": rng.uniform(0.65, 1.6) if rng.random() < 0.7 else 1.0,
            "sy": rng.uniform(0.65, 1.4) if rng.random() < 0.7 else 1.0,
            "rot": rng.uniform(-20, 20) if rng.random() < 0.6 else 0.0,
            "sh": rng.uniform(-0.25, 0.25) if rng.random() < 0.4 else 0.0,
            "mir": -1.0 if rng.random() < 0.15 else 1.0,
            "ox": rng.uniform(-0.3, 0.3), "oy": rng.uniform(-0.3, 0.3)}


def _lpt(x, y, P, cx, cy):
    x *= P["mir"]
    x *= P["sx"]
    y *= P["sy"]
    x += P["sh"] * y
    if P["rot"]:
        a = math.radians(P["rot"])
        x, y = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
    return (cx + x, cy + y)


def _ngon(lcx, lcy, rx, ry, n, rot0, P):
    pts = []
    for i in range(n):
        a = math.radians(rot0 + i * 360.0 / n)
        pts.append(_lpt(rx * math.cos(a), ry * math.sin(a), P, lcx, lcy))
    return pts


def _band(lcx, lcy, ro, ri, a0, a1, n, P):
    pts = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        pts.append(_lpt(ro * math.cos(a), ro * math.sin(a), P, lcx, lcy))
    for i in range(n + 1):
        a = math.radians(a1 - (a1 - a0) * i / n)
        pts.append(_lpt(ri * math.cos(a), ri * math.sin(a), P, lcx, lcy))
    return pts


def _opoly(td, pts, fill, pal, lw_mul=1):
    """Polygon with the logo's outline treatment (single / double / none)."""
    oc, ow = pal["oc"], max(1, int(round(pal["ow"] * lw_mul)))
    f = None if fill is None else (fill + (255,) if len(fill) == 3 else fill)
    if pal["double"] and fill is not None:
        td.polygon(pts, fill=(oc + (255,)) if len(oc) == 3 else oc)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pts = [(cx + (x - cx) * 0.78, cy + (y - cy) * 0.78) for (x, y) in pts]
        td.polygon(pts, fill=f)
        return
    if oc is None:
        if f is not None:
            td.polygon(pts, fill=f)
    else:
        td.polygon(pts, fill=f, outline=oc + (255,) if len(oc) == 3 else oc, width=ow)


def _hstripes(td, pts, lcx, lcy, R, color, pal):
    """Inset horizontal bars clipped (approximately) to a shape's bbox."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    m = pal["ow"] + 4
    x0, x1, y0, y1 = min(xs) + m, max(xs) - m, min(ys) + m, max(ys) - m
    if x1 <= x0 or y1 <= y0:
        return
    k = 4
    for i in range(k):
        y = y0 + (y1 - y0) * (i + 0.5) / k
        taper = 1.0 - 0.35 * abs(y - lcy) / max(1.0, (y1 - y0) / 2)
        cxm = (x0 + x1) / 2
        hw = (x1 - x0) / 2 * max(0.3, taper)
        hh = (y1 - y0) / k * 0.22
        td.rectangle([cxm - hw, y - hh, cxm + hw, y + hh], fill=color + (255,))


def _comp_tile(dst, src, dx, dy):
    """alpha_composite with clipping (PIL raises on out-of-bounds dest)."""
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(dst.width, dx + src.width), min(dst.height, dy + src.height)
    if x1 <= x0 or y1 <= y0:
        return
    dst.alpha_composite(src.crop((x0 - dx, y0 - dy, x1 - dx, y1 - dy)), (x0, y0))


def _stamp_text(tile, s, font, fill, soc, sw, x, y, tilt):
    """Text stamp with optional raster tilt (real logo letters rarely sit upright)."""
    bb = font.getbbox(s)
    w, h = bb[2] - bb[0] + sw * 2 + 8, bb[3] - bb[1] + sw * 2 + 8
    t = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    td = ImageDraw.Draw(t)
    f4 = fill + (255,) if len(fill) == 3 else fill
    if soc is None:
        td.text((sw + 4 - bb[0], sw + 4 - bb[1]), s, font=font, fill=f4)
    else:
        soc_f = _flat_stroke(soc)
        s4 = soc_f + (255,) if len(soc_f) == 3 else soc_f
        td.text((sw + 4 - bb[0], sw + 4 - bb[1]), s, font=font, fill=f4,
                stroke_width=sw, stroke_fill=s4)
    if tilt:
        t = t.rotate(tilt, expand=True, resample=Image.BICUBIC)
    _comp_tile(tile, t, int(x - t.width / 2), int(y - t.height / 2))


def _logo_backing(tile, td, lcx, lcy, R, pal, rng, P):
    style = rng.choices(["none", "disc", "ring", "shield", "hex", "blob", "ribbon",
                         "chamfer", "seal", "ticket", "wavy"],
                        weights=[26, 17, 10, 10, 7, 8, 7,
                                 5, 4, 3, 3])[0]
    if style == "none":
        return
    b = pal["back"] + (255,)
    thin = dict(pal)
    thin.update({"ow": max(1, pal["ow"] - 1), "double": False,
                 "shadow": None, "glow": None})
    pad = R * rng.uniform(0.25, 0.5)
    if style == "disc":
        _opoly(td, _ngon(lcx, lcy, R + pad, R + pad, 26, rng.uniform(0, 90), P), b, thin)
    elif style == "ring":
        _opoly(td, _band(lcx, lcy, R + pad, R + pad - max(3, R * 0.18), 0, 360, 26, P), b, thin)
    elif style == "shield":
        w, h = (R + pad) * 0.95, R + pad
        pts = [_lpt(x, y, P, lcx, lcy) for (x, y) in
               [(-w, -h), (w, -h), (w, h * 0.1), (0, h), (-w, h * 0.1)]]
        _opoly(td, pts, b, thin)
    elif style == "hex":
        _opoly(td, _ngon(lcx, lcy, R + pad, (R + pad) * 0.92, 6, rng.uniform(0, 60), P), b, thin)
    elif style == "blob":
        raw = _blob_pts(0, 0, R + pad, (R + pad) * 0.9, rng.randint(7, 10),
                        rng.uniform(0.08, 0.2), rng)
        _opoly(td, [_lpt(x, y, P, lcx, lcy) for (x, y) in raw], b, thin)
    elif style == "chamfer":
        w, h = R + pad, (R + pad) * 0.9
        c = min(w, h) * rng.uniform(0.18, 0.35)
        raw = [(-w + c, -h), (w - c, -h), (w, -h + c), (w, h - c),
               (w - c, h), (-w + c, h), (-w, h - c), (-w, -h + c)]
        _opoly(td, [_lpt(x, y, P, lcx, lcy) for (x, y) in raw], b, thin)
    elif style == "seal":
        raw = _seal_pts(0, 0, R + pad, (R + pad) * 0.9, rng)
        _opoly(td, [_lpt(x, y, P, lcx, lcy) for (x, y) in raw], b, thin)
    elif style == "ticket":
        w, h = R + pad, (R + pad) * 0.72
        nr = min(w, h) * 0.22
        raw = [(-w, -h), (w, -h), (w, -nr)] + \
            [(w - nr + nr * math.cos(-math.pi / 2 + math.pi * i / 6),
              nr * math.sin(-math.pi / 2 + math.pi * i / 6)) for i in range(1, 6)] + \
            [(w, nr), (w, h), (-w, h), (-w, nr)] + \
            [(-w + nr + nr * math.cos(math.pi / 2 + math.pi * i / 6),
              nr * math.sin(math.pi / 2 + math.pi * i / 6)) for i in range(1, 6)] + \
            [(-w, -nr)]
        _opoly(td, [_lpt(x, y, P, lcx, lcy) for (x, y) in raw], b, thin)
    elif style == "wavy":
        raw = _wavy_rect_pts((R + pad) * 2, (R + pad) * 1.8, rng, m=0.0, n=6)
        cxm = sum(p[0] for p in raw) / len(raw)
        cym = sum(p[1] for p in raw) / len(raw)
        raw = [(x - cxm, y - cym) for (x, y) in raw]
        _opoly(td, [_lpt(x, y, P, lcx, lcy) for (x, y) in raw], b, thin)
    else:  # ribbon: straight bar + folded darker tail on one side
        side = rng.choice([-1, 1])
        bw, bh = (R + pad) * 1.5, (R + pad) * 0.62
        x0, x1 = lcx - bw + side * R * 0.5, lcx + bw + side * R * 0.5
        td.rectangle([x0, lcy - bh, x1, lcy + bh], fill=b)
        tail = side * (bw + R * 0.55)
        edge = x1 if side > 0 else x0
        td.polygon([(edge, lcy - bh), (edge + tail, lcy),
                    (edge, lcy + bh)], fill=_shade(pal["back"], 0.7) + (255,))


def _logo_micro(tile, td, lcx, lcy, R, pal, rng, P):
    for _ in range(rng.choices([0, 1, 2], weights=[35, 45, 20])[0]):
        m = rng.choice(["spark", "dots", "bar", "tm", "gloss"])
        ax = lcx + rng.uniform(-1.1, 1.1) * R
        ay = lcy + rng.uniform(-1.1, 1.1) * R
        acc = pal["c2"] + (255,)
        if m == "spark":
            s = R * rng.uniform(0.10, 0.2)
            td.polygon([(ax, ay - s), (ax + s * 0.25, ay - s * 0.25), (ax + s, ay),
                        (ax + s * 0.25, ay + s * 0.25), (ax, ay + s),
                        (ax - s * 0.25, ay + s * 0.25), (ax - s, ay),
                        (ax - s * 0.25, ay - s * 0.25)], fill=acc)
        elif m == "dots":
            for k in range(3):
                rr = R * 0.07
                td.ellipse([ax + k * R * 0.22 - rr, ay - rr,
                            ax + k * R * 0.22 + rr, ay + rr], fill=acc)
        elif m == "bar":
            w2, hh = R * rng.uniform(0.3, 0.6), max(2, R * 0.08)
            td.rectangle([ax - w2, ay - hh, ax + w2, ay + hh], fill=acc)
        elif m == "tm":
            f = load_font_for("TM", max(8, int(R * 0.28)))
            td.text((ax, ay), "TM", font=f, fill=(245, 245, 245, 255))
        else:
            rr = R * rng.uniform(0.08, 0.14)
            td.ellipse([ax - rr, ay - rr, ax + rr, ay + rr], fill=(255, 255, 255, 220))


def _lg_monogram(tile, td, lcx, lcy, R, pal, rng, P):
    ch = "".join(rng.choice(_LOGO_GLYPHS) for _ in range(1 if rng.random() < 0.7 else 2))
    frame = rng.choice(["disc", "ring", "shield", "hex", "none"])
    fm = rng.choice(["solid", "nested", "hollow"])
    if frame == "disc":
        if fm == "hollow":
            td.polygon(_ngon(lcx, lcy, R, R, 26, 0, P), fill=None,
                       outline=(pal["oc"] or pal["c1"]) + (255,), width=max(2, pal["ow"] + 1))
        else:
            _opoly(td, _ngon(lcx, lcy, R, R, 26, 0, P), pal["c1"], pal)
            if fm == "nested":
                _opoly(td, _ngon(lcx, lcy + R * 0.08, R * 0.62, R * 0.55, 22, 0, P),
                       _lerp3(pal["c1"], pal["c2"], 0.5),
                       {**pal, "double": False, "oc": None})
    elif frame == "ring":
        _opoly(td, _band(lcx, lcy, R, R * 0.72, 0, 360, 26, P), pal["c1"], pal)
    elif frame == "shield":
        w, h = R * 0.9, R
        _opoly(td, [_lpt(x, y, P, lcx, lcy) for (x, y) in
                    [(-w, -h), (w, -h), (w, 0), (0, h), (-w, 0)]], pal["c1"], pal)
    elif frame == "hex":
        _opoly(td, _ngon(lcx, lcy, R, R * 0.9, 6, rng.uniform(0, 60), P), pal["c1"], pal)
    ink = pal["c2"]
    if frame != "none" and rng.random() < 0.4:
        ink = (245, 245, 245) if sum(pal["c1"]) < 380 else (15, 15, 15)
    size = int(R * rng.uniform(0.9, 1.35)) if len(ch) == 1 else int(R * 0.8)
    tilt = rng.uniform(-18, 18) if rng.random() < 0.4 else 0
    _stamp_text(tile, ch, load_font_for(ch, size), ink, pal["oc"], max(1, pal["ow"] - 1),
                lcx + R * rng.uniform(-0.12, 0.12), lcy, tilt)


def _lg_duogram(tile, td, lcx, lcy, R, pal, rng, P):
    n = rng.randint(2, 3)
    chs = "".join(rng.choice(_LOGO_GLYPHS) for _ in range(n))
    if rng.random() < 0.5:
        _opoly(td, _band(lcx, lcy, R * 1.05, R * 0.85, 0, 360, 26, P), pal["c1"], pal)
    size = int(R * rng.uniform(0.75, 0.95))
    mode = rng.choice(["same", "alt", "lerp"])
    total = size * (0.55 + 0.6 * (n - 1) * rng.uniform(0.5, 0.7))
    x = lcx - total / 2
    for i, ch in enumerate(chs):
        if mode == "same":
            f = pal["c1"]
        elif mode == "alt":
            f = pal["c1"] if i % 2 == 0 else pal["c2"]
        else:
            f = _lerp3(pal["c1"], pal["c2"], i / max(1, n - 1))
        yj = lcy + (i - (n - 1) / 2) * size * 0.12 * rng.uniform(-1, 1)
        tilt = rng.uniform(-14, 14) if rng.random() < 0.5 else 0
        _stamp_text(tile, ch, load_font_for(ch, size), f, pal["oc"], max(1, pal["ow"] - 1),
                    x + size * 0.3, yj, tilt)
        x += size * rng.uniform(0.52, 0.68)
    if rng.random() < 0.5:
        w2 = total * rng.uniform(0.4, 0.6)
        ox = R * rng.uniform(-0.2, 0.2)
        uh = max(2, R * 0.09)
        td.rectangle([lcx + ox - w2, lcy + size * 0.55, lcx + ox + w2,
                      lcy + size * 0.55 + uh], fill=pal["c2"] + (255,))
        if rng.random() < 0.3:
            td.rectangle([lcx + ox - w2, lcy + size * 0.55 + uh + 2,
                          lcx + ox + w2, lcy + size * 0.55 + uh * 2 + 2],
                         fill=pal["c1"] + (255,))


def _lg_burst(tile, td, lcx, lcy, R, pal, rng, P):
    n = rng.randint(7, 14)
    ro = R * rng.uniform(0.9, 1.15)
    ratio = rng.uniform(0.55, 0.85)
    rot = rng.uniform(0, 360)
    fm = rng.choice(["solid", "nested", "hollow", "striped"])
    pts = [_lpt(x, y, P, lcx, lcy) for (x, y) in _star_points(0, 0, ro, ro * ratio, n, rot)]
    if fm == "hollow":
        td.polygon(pts, fill=None, outline=(pal["oc"] or pal["c1"]) + (255,),
                   width=max(2, pal["ow"] + 1))
    else:
        _opoly(td, pts, pal["c1"], pal)
        if fm == "nested":
            pts2 = [_lpt(x * 0.62, y * 0.62, P, lcx, lcy)
                    for (x, y) in _star_points(0, 0, ro, ro * ratio, n, rot)]
            _opoly(td, pts2, _lerp3(pal["c1"], pal["c2"], 0.55),
                   {**pal, "double": False, "oc": None})
        elif fm == "striped":
            _hstripes(td, pts, lcx, lcy, R, pal["c2"], pal)
    core = rng.random()
    if core < 0.35:
        rr = R * rng.uniform(0.10, 0.18)
        ox, oy = R * rng.uniform(-0.15, 0.15), R * rng.uniform(-0.15, 0.15)
        td.ellipse([lcx + ox - rr, lcy + oy - rr, lcx + ox + rr, lcy + oy + rr],
                   fill=pal["c2"] + (255,))
    elif core < 0.5:
        ch = rng.choice(_LOGO_GLYPHS)
        ink = (245, 245, 245) if sum(pal["c1"]) < 380 else (15, 15, 15)
        _stamp_text(tile, ch, load_font_for(ch, int(R * 0.45)), ink, None, 1,
                    lcx + R * rng.uniform(-0.1, 0.1), lcy, 0)


def _lg_starbadge(tile, td, lcx, lcy, R, pal, rng, P):
    _opoly(td, _band(lcx, lcy, R, R * 0.78, 0, 360, 26, P), pal["c1"], pal)
    sp = [_lpt(x, y, P, lcx, lcy) for (x, y) in
          _star_points(0, 0, R * 0.62, R * 0.26, 5, rng.uniform(0, 360))]
    _opoly(td, sp, pal["c2"], {**pal, "double": False})
    rr = R * rng.uniform(0.08, 0.14)
    ox, oy = R * rng.uniform(-0.12, 0.12), R * rng.uniform(-0.12, 0.12)
    if rng.random() < 0.7:
        td.ellipse([lcx + ox - rr, lcy + oy - rr, lcx + ox + rr, lcy + oy + rr],
                   fill=pal["c1"] + (255,))
    else:
        td.polygon([(lcx + ox, lcy + oy - rr * 1.4), (lcx + ox + rr, lcy + oy),
                    (lcx + ox, lcy + oy + rr * 1.4), (lcx + ox - rr, lcy + oy)],
                   fill=pal["c1"] + (255,))


def _lg_orb(tile, td, lcx, lcy, R, pal, rng, P):
    _opoly(td, _ngon(lcx, lcy, R, R, 26, 0, P), pal["c1"], pal)
    steps = rng.randint(2, 3)
    hx, hy = rng.uniform(-0.4, 0.4), rng.uniform(-0.5, -0.1)
    for i in range(steps, 0, -1):
        t = i / (steps + 1)
        col = _lerp3(pal["c1"], pal["c2"], 1 - t * 0.8)
        rr = R * (0.35 + 0.45 * t)
        _opoly(td, _ngon(lcx + hx * R * t, lcy + hy * R * t, rr, rr * 0.92, 22, 0, P),
               col, {**pal, "double": False, "oc": None})
    gl = rng.random()
    if gl < 0.35:
        gx, gy = lcx + R * rng.uniform(-0.45, 0.45), lcy - R * rng.uniform(0.3, 0.55)
        gr = R * rng.uniform(0.10, 0.2)
        td.ellipse([gx - gr, gy - gr * 0.7, gx + gr, gy + gr * 0.7],
                   fill=(255, 255, 255, 235))
    elif gl < 0.5:
        gx = lcx + R * rng.uniform(-0.3, 0.3)
        gy, gw = lcy - R * 0.45, R * rng.uniform(0.2, 0.4)
        td.line([(gx - gw, gy), (gx + gw, gy)], fill=(255, 255, 255, 220),
                width=max(2, int(R * 0.08)))


def _lg_bolt(tile, td, lcx, lcy, R, pal, rng, P):
    lean = rng.uniform(-0.35, 0.35)
    w = R * rng.uniform(0.35, 0.6)
    pts = [_lpt(x, y, P, lcx, lcy) for (x, y) in
           [(lean * R + w * 0.5, -R), (-w * 0.6, R * 0.15), (-w * 0.05, R * 0.15),
            (lean * R - w * 0.5, R), (w * 0.6, -R * 0.15), (w * 0.05, -R * 0.15)]]
    fm = rng.choice(["solid", "nested", "hollow", "duo"])
    if fm == "duo":
        ex, ey = R * rng.uniform(0.15, 0.35) * rng.choice([-1, 1]), R * rng.uniform(-0.15, 0.2)
        echo = [(x + ex, y + ey) for (x, y) in pts]
        td.polygon(echo, fill=pal["c2"] + (255,))
        _opoly(td, pts, pal["c1"], pal)
    elif fm == "hollow":
        td.polygon(pts, fill=None, outline=(pal["oc"] or pal["c1"]) + (255,),
                   width=max(2, pal["ow"] + 1))
    else:
        _opoly(td, pts, pal["c1"], pal)
        if fm == "nested":
            core = [(lcx + (x - lcx) * 0.45, lcy + (y - lcy) * 0.55) for (x, y) in pts]
            td.polygon(core, fill=pal["c2"] + (255,))


def _lg_rings(tile, td, lcx, lcy, R, pal, rng, P):
    n = rng.randint(2, 3)
    form = rng.choice(["line", "tri", "vee"])
    cols = [pal["c1"], pal["c2"], _lerp3(pal["c1"], pal["c2"], 0.5)]
    if form == "line":
        spots = [(-R * 0.5, 0), (R * 0.5, 0), (0, -R * 0.55)][:n]
    elif form == "vee":
        spots = [(-R * 0.5, -R * 0.3), (R * 0.5, -R * 0.3), (0, R * 0.45)][:n]
    else:
        spots = [(-R * 0.45, R * 0.25), (R * 0.45, R * 0.25), (0, -R * 0.45)][:n]
    wdt = rng.uniform(0.10, 0.2)
    for i in range(n):
        ox, oy = spots[i]
        rr = R * rng.uniform(0.38, 0.55)
        if rng.random() < 0.2 and i == n - 1:
            _opoly(td, _ngon(lcx + ox, lcy + oy, rr, rr, 22, 0, P),
                   cols[i], {**pal, "double": False})
        else:
            _opoly(td, _band(lcx + ox, lcy + oy, rr, rr * (1 - wdt * 2), 0, 360, 22, P),
                   cols[i], {**pal, "double": False, "oc": None})


def _lg_drop(tile, td, lcx, lcy, R, pal, rng, P):
    tip = R * rng.uniform(0.9, 1.35)
    lean = rng.uniform(-0.35, 0.35)
    w = R * rng.uniform(0.62, 0.78)
    pts = [_lpt(x, y, P, lcx, lcy) for (x, y) in
           [(lean * tip, -tip), (w, -R * 0.1), (w * 0.7, R * 0.55),
            (0, R * 0.85), (-w * 0.7, R * 0.55), (-w, -R * 0.1)]]
    fm = rng.choice(["solid", "nested", "hollow", "striped"])
    if fm == "hollow":
        td.polygon(pts, fill=None, outline=(pal["oc"] or pal["c1"]) + (255,),
                   width=max(2, pal["ow"] + 1))
    else:
        _opoly(td, pts, pal["c1"], pal)
        if fm == "nested":
            inner = [(lcx + (x - lcx) * 0.55, lcy + (y - lcy) * 0.55 + R * 0.1)
                     for (x, y) in pts]
            td.polygon(inner, fill=pal["c2"] + (255,))
        elif fm == "striped":
            _hstripes(td, pts, lcx, lcy, R, pal["c2"], pal)
    hl = rng.random()
    if hl < 0.4:
        sx = lcx + rng.choice([-1, 1]) * R * rng.uniform(0.15, 0.35)
        y1, y2 = lcy - R * rng.uniform(0.35, 0.5), lcy + R * rng.uniform(0.25, 0.4)
        td.line([(sx, y1), (sx, y2)], fill=(255, 255, 255, 230), width=max(2, int(R * 0.12)))
    elif hl < 0.55:
        gx, gy = lcx + R * rng.uniform(-0.3, 0.3), lcy - R * rng.uniform(0.2, 0.45)
        gr = R * rng.uniform(0.08, 0.15)
        td.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=(255, 255, 255, 235))


def _lg_chevron(tile, td, lcx, lcy, R, pal, rng, P):
    n = rng.randint(2, 4)
    direction = rng.choice(["up", "right", "down"])
    th = R * rng.uniform(0.16, 0.28)
    gap = R * rng.uniform(0.28, 0.5)
    stag = R * rng.uniform(0.0, 0.3) if rng.random() < 0.5 else 0.0
    mode = rng.choice(["alt", "lerp", "same"])
    span = R * 0.85
    y0 = lcy - gap * (n - 1) / 2
    for i in range(n):
        if mode == "same":
            col = pal["c1"]
        elif mode == "alt":
            col = pal["c1"] if i % 2 == 0 else pal["c2"]
        else:
            col = _lerp3(pal["c1"], pal["c2"], i / max(1, n - 1))
        base = [(-span, th), (0, -th), (span, th),
                (span, th + th * 1.4), (0, -th + th * 1.4), (-span, th + th * 1.4)]
        if direction == "right":
            base = [(-y, x) for (x, y) in base]
        elif direction == "down":
            base = [(-x, -y) for (x, y) in base]
        pts = [_lpt(x, y, P, lcx + i * stag, y0 + i * gap) for (x, y) in base]
        _opoly(td, pts, col, {**pal, "double": False})


def _lg_shield(tile, td, lcx, lcy, R, pal, rng, P):
    w, h = R * rng.uniform(0.8, 1.0), R
    shoulder = rng.choice(["straight", "notch", "round"])
    top = [(-w, -h), (w, -h)]
    if shoulder == "straight":
        raw = top + [(w, h * 0.1), (0, h), (-w, h * 0.1)]
    elif shoulder == "notch":
        raw = top + [(w, -h * 0.4), (w * 0.7, -h * 0.25), (w * 0.7, h * 0.1),
                     (0, h), (-w * 0.7, h * 0.1),
                     (-w * 0.7, -h * 0.25), (-w, -h * 0.4)]
    else:
        raw = top + [(w * 1.05, -h * 0.2), (w * 0.8, h * 0.25), (0, h),
                     (-w * 0.8, h * 0.25), (-w * 1.05, -h * 0.2)]
    pts = [_lpt(x, y, P, lcx, lcy) for (x, y) in raw]
    fm = rng.choice(["solid", "nested", "striped"])
    _opoly(td, pts, pal["c1"], pal)
    if fm == "nested":
        inner = [(lcx + (x - lcx) * 0.62, lcy + (y - lcy) * 0.62) for (x, y) in pts]
        td.polygon(inner, fill=pal["c2"] + (255,))
    elif fm == "striped":
        _hstripes(td, pts, lcx, lcy, R, pal["c2"], pal)
    band = rng.choices(["top", "bottom", "sash", "none"], weights=[35, 20, 20, 25])[0]
    if band == "top":
        td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                    [(-w * 0.98, -h * 0.98), (w * 0.98, -h * 0.98),
                     (w * 0.98, -h * 0.45), (-w * 0.98, -h * 0.45)]],
                   fill=pal["c2"] + (255,))
    elif band == "bottom":
        td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                    [(-w * 0.5, h * 0.25), (w * 0.5, h * 0.25),
                     (w * 0.5, h * 0.5), (-w * 0.5, h * 0.5)]],
                   fill=pal["c2"] + (255,))
    elif band == "sash":
        td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                    [(-w * 0.5, -h * 0.9), (-w * 0.05, -h * 0.9),
                     (w * 0.25, h * 0.3), (-w * 0.2, h * 0.3)]],
                   fill=pal["c2"] + (255,))
    if rng.random() < 0.75:
        _inner_device(tile, td, lcx, lcy + R * 0.15, R * 0.55, pal, rng, P,
                      (245, 245, 245) if sum(pal["c1"]) < 380 else (15, 15, 15),
                      pal["c2"])


def _lg_bubble(tile, td, lcx, lcy, R, pal, rng, P):
    w, h = R * rng.uniform(0.9, 1.2), R * rng.uniform(0.6, 0.85)
    nch = rng.randint(1, 3)
    chs = "".join(rng.choice(_LOGO_GLYPHS) for _ in range(nch))
    tail = rng.choice(["bl", "br", "tr", "tl", "l", "r"])
    cy0 = lcy - R * 0.1
    _opoly(td, [_lpt(x, y, P, lcx, cy0) for (x, y) in
                [(-w, -h), (w, -h), (w, h), (-w, h)]], pal["c1"], pal)
    tails = {"bl": [(-w * 0.5, h), (-w * 0.9, h + R * 0.7), (-w * 0.1, h)],
             "br": [(w * 0.5, h), (w * 0.9, h + R * 0.7), (w * 0.1, h)],
             "tr": [(w * 0.5, -h), (w * 0.9, -h - R * 0.7), (w * 0.1, -h)],
             "tl": [(-w * 0.5, -h), (-w * 0.9, -h - R * 0.7), (-w * 0.1, -h)],
             "l": [(-w, -h * 0.3), (-w - R * 0.7, 0), (-w, h * 0.3)],
             "r": [(w, -h * 0.3), (w + R * 0.7, 0), (w, h * 0.3)]}[tail]
    td.polygon([_lpt(x, y, P, lcx, cy0) for (x, y) in tails], fill=pal["c1"] + (255,))
    ink = (245, 245, 245) if sum(pal["c1"]) < 380 else (15, 15, 15)
    _stamp_text(tile, chs, load_font_for(chs, int(h * rng.uniform(0.85, 1.05))), ink, None, 1,
                lcx, cy0, rng.uniform(-10, 10) if rng.random() < 0.4 else 0)


def _lg_crown(tile, td, lcx, lcy, R, pal, rng, P):
    n = rng.randint(3, 5)
    w = R * rng.uniform(0.85, 1.05)
    base_y, tip_y = R * 0.55, -R * rng.uniform(0.7, 1.0)
    raw = [(-w, base_y)]
    for i in range(2 * n - 1):
        x = -w + 2 * w * i / (2 * n - 2)
        y = tip_y * rng.uniform(0.85, 1.1) if i % 2 == 1 else -R * 0.1
        raw.append((x, y))
    raw += [(w, base_y)]
    _opoly(td, [_lpt(x, y, P, lcx, lcy) for (x, y) in raw], pal["c1"], pal)
    bandh = R * 0.28
    td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                [(-w, base_y - bandh), (w, base_y - bandh), (w, base_y), (-w, base_y)]],
               fill=pal["c2"] + (255,))
    ng = rng.randint(1, 3)
    gfill = rng.choice([(255, 255, 255, 235), pal["c2"] + (255,), (20, 20, 20, 255)])
    gshape = rng.choice(["dot", "diamond"])
    for k in range(ng):
        gx = (k - (ng - 1) / 2) * (w / max(1, ng)) * 1.2
        gy = base_y - bandh / 2 + R * rng.uniform(-0.05, 0.05)
        gr = R * rng.uniform(0.07, 0.11)
        if gshape == "dot":
            td.ellipse([lcx + gx - gr, lcy + gy - gr, lcx + gx + gr, lcy + gy + gr],
                       fill=gfill)
        else:
            td.polygon([(lcx + gx, lcy + gy - gr * 1.3), (lcx + gx + gr, lcy + gy),
                        (lcx + gx, lcy + gy + gr * 1.3), (lcx + gx - gr, lcy + gy)],
                       fill=gfill)


def _lg_spiral(tile, td, lcx, lcy, R, pal, rng, P):
    turns = rng.uniform(1.5, 2.5)
    a0 = rng.uniform(0, 360)
    cw = rng.choice([-1, 1])
    segs = 26
    th = R * rng.uniform(0.10, 0.18)
    prev = None
    for i in range(segs):
        t = i / (segs - 1)
        a = math.radians(a0 + cw * turns * 360 * t)
        rr = R * (0.15 + 0.85 * t)
        p = _lpt(rr * math.cos(a), rr * math.sin(a), P, lcx, lcy)
        if prev is not None:
            dx, dy = p[0] - prev[0], p[1] - prev[1]
            L = max(1.0, math.hypot(dx, dy))
            nx, ny = -dy / L * th, dx / L * th
            col = pal["c1"] if (i % 2 == 0 or rng.random() < 0.6) else pal["c2"]
            td.polygon([(prev[0] + nx, prev[1] + ny), (p[0] + nx, p[1] + ny),
                        (p[0] - nx, p[1] - ny), (prev[0] - nx, prev[1] - ny)],
                       fill=col + (255,))
        prev = p
    cc = rng.random()
    if cc < 0.5:
        er = R * rng.uniform(0.09, 0.15)
        ox, oy = R * rng.uniform(-0.08, 0.08), R * rng.uniform(-0.08, 0.08)
        td.ellipse([lcx + ox - er, lcy + oy - er, lcx + ox + er, lcy + oy + er],
                   fill=pal["c2"] + (255,))
    elif cc < 0.7:
        er = R * 0.16
        td.ellipse([lcx - er, lcy - er, lcx + er, lcy + er], fill=None,
                   outline=pal["c2"] + (255,), width=max(2, int(R * 0.08)))


def _lg_cube(tile, td, lcx, lcy, R, pal, rng, P):
    flip = rng.choice([-1, 1])
    w, hu, hd = R * 0.8, R * 0.45, R * 0.55
    cy0 = lcy - R * 0.15
    thin = {**pal, "double": False}
    left = [_lpt(x * flip, y, P, lcx, cy0) for (x, y) in
            [(-w, 0), (0, hu), (0, hu + hd), (-w, hd)]]
    right = [_lpt(x * flip, y, P, lcx, cy0) for (x, y) in
             [(w, 0), (0, hu), (0, hu + hd), (w, hd)]]
    top = [_lpt(x * flip, y, P, lcx, cy0) for (x, y) in
           [(0, -hu), (w, 0), (0, hu), (-w, 0)]]
    _opoly(td, left, _shade(pal["c1"], 0.55), thin)
    _opoly(td, right, _shade(pal["c1"], 0.8), thin)
    _opoly(td, top, _lerp3(pal["c1"], (255, 255, 255), 0.3), thin)
    if rng.random() < 0.3:
        s = rng.uniform(0.28, 0.42)
        side = rng.choice([-1, 1])
        ox, oy = side * R * rng.uniform(0.75, 1.0), -R * rng.uniform(0.55, 0.85)
        _opoly(td, [_lpt(x * s + ox, y * s + oy, P, lcx, lcy) for (x, y) in
                    [(0, -hu), (w, 0), (0, hu), (-w, 0)]], pal["c2"], thin)


def _lg_waves(tile, td, lcx, lcy, R, pal, rng, P):
    n = rng.randint(3, 5)
    vertical = rng.random() < 0.25
    mode = rng.choice(["alt", "lerp"])
    gap = rng.uniform(0.32, 0.45)
    for i in range(n):
        t = i / max(1, n - 1)
        col = (pal["c1"] if i % 2 == 0 else pal["c2"]) if mode == "alt" \
            else _lerp3(pal["c1"], pal["c2"], t)
        off = (i - (n - 1) / 2) * R * gap
        amp = R * rng.uniform(0.08, 0.22)
        ph = rng.uniform(0, 6.28)
        ln = rng.uniform(0.7, 1.0) * R
        th = R * rng.uniform(0.10, 0.18)
        raw = []
        for k in range(13):
            u = -ln + 2 * ln * k / 12
            v = off + amp * math.sin(ph + u / max(1.0, R) * 3.0)
            raw.append((u, v - th) if not vertical else (v - th, u))
        for k in range(12, -1, -1):
            u = -ln + 2 * ln * k / 12
            v = off + amp * math.sin(ph + u / max(1.0, R) * 3.0)
            raw.append((u, v + th) if not vertical else (v + th, u))
        td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in raw], fill=col + (255,))


def _lg_ribbon(tile, td, lcx, lcy, R, pal, rng, P):
    w, h = R * rng.uniform(1.0, 1.3), R * rng.uniform(0.32, 0.5)
    ends = rng.choice(["both", "left", "right", "notch"])
    _opoly(td, [_lpt(x, y, P, lcx, lcy) for (x, y) in
                [(-w, -h), (w, -h), (w, h), (-w, h)]], pal["c1"], pal)
    fd = _shade(pal["c1"], 0.6) + (255,)
    if ends in ("both", "left"):
        td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                    [(-w, -h), (-w - R * 0.5, -h * 0.4), (-w, 0),
                     (-w - R * 0.5, h * 0.4), (-w, h)]], fill=fd)
    if ends in ("both", "right"):
        td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                    [(w, -h), (w + R * 0.5, -h * 0.4), (w, 0),
                     (w + R * 0.5, h * 0.4), (w, h)]], fill=fd)
    if ends == "notch":
        td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                    [(w, -h), (w - R * 0.4, 0), (w, h)]], fill=fd)
    rc = rng.random()
    ink = (245, 245, 245) if sum(pal["c1"]) < 380 else (15, 15, 15)
    jx = R * rng.uniform(-0.15, 0.15)
    if rc < 0.45:
        ch = rng.choice(_LOGO_GLYPHS)
        _stamp_text(tile, ch, load_font_for(ch, int(h * 1.1)), ink, None, 1, lcx + jx, lcy, 0)
    elif rc < 0.65:
        chs = "".join(rng.choice(_LOGO_GLYPHS) for _ in range(2))
        _stamp_text(tile, chs, load_font_for(chs, int(h * 0.8)), ink, None, 1, lcx + jx, lcy, 0)
    elif rc < 0.85:
        rr = h * rng.uniform(0.22, 0.34)
        td.ellipse([lcx + jx - rr, lcy - rr, lcx + jx + rr, lcy + rr], fill=pal["c2"] + (255,))


def _lg_torii(tile, td, lcx, lcy, R, pal, rng, P):
    """Torii-gate mark: twin pillars + double lintel (scanlator gate family)."""
    w = R * rng.uniform(0.7, 0.9)
    top_y, bot_y = lcy - R * 0.7, lcy + R * 0.8
    curve = rng.uniform(0, R * 0.15)
    # top lintel (curved horns) + second beam
    td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                [(-w - R * 0.25, -R * 0.95 + curve), (w + R * 0.25, -R * 0.95 + curve),
                 (w + R * 0.25, -R * 0.65), (-w - R * 0.25, -R * 0.65)]],
               fill=pal["c1"] + (255,))
    td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                [(-w, -R * 0.45), (w, -R * 0.45), (w, -R * 0.2), (-w, -R * 0.2)]],
               fill=pal["c2"] + (255,))
    for sx in (-1, 1):
        td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in
                    [(sx * w * 0.75 - R * 0.12, -R * 0.65), (sx * w * 0.75 + R * 0.12, -R * 0.65),
                     (sx * w * 0.75 + R * 0.12, R * 0.85), (sx * w * 0.75 - R * 0.12, R * 0.85)]],
                   fill=pal["c1"] + (255,))
    if rng.random() < 0.5:
        ch = rng.choice(_LOGO_GLYPHS)
        _stamp_text(tile, ch, load_font_for(ch, int(R * 0.4)),
                    (245, 245, 255) if sum(pal["c1"]) < 380 else (15, 15, 15),
                    None, 1, lcx, lcy + R * 0.35, 0)


def _lg_linked(tile, td, lcx, lcy, R, pal, rng, P):
    """Interlocked rings: 2-3 overlapping circles with alternating overpass."""
    n = rng.randint(2, 3)
    offs = [(-R * 0.4, 0), (R * 0.4, 0), (0, -R * 0.45)][:n]
    cols = [pal["c1"], pal["c2"], _lerp3(pal["c1"], pal["c2"], 0.5)]
    wdt = max(2, int(R * rng.uniform(0.12, 0.20)))
    for i, (ox, oy) in enumerate(offs):
        rr = R * rng.uniform(0.42, 0.55)
        _opoly(td, _band(lcx + ox, lcy + oy, rr, rr - wdt, 0, 360, 24, P),
               cols[i], {**pal, "double": False, "oc": None})
    # overpass notch: redraw a small arc of ring 0 on top
    td.arc([lcx + offs[0][0] - R * 0.5, lcy + offs[0][1] - R * 0.5,
            lcx + offs[0][0] + R * 0.5, lcy + offs[0][1] + R * 0.5],
           start=rng.uniform(0, 360), end=rng.uniform(0, 360) + 70,
           fill=cols[0] + (255,), width=wdt + 2)


def _lg_speechmono(tile, td, lcx, lcy, R, pal, rng, P):
    """Speech-tail monogram: rounded box + tail + 1-2 letters (compact seal)."""
    w, h = R * rng.uniform(0.8, 1.0), R * rng.uniform(0.55, 0.75)
    _opoly(td, [_lpt(x, y, P, lcx, lcy) for (x, y) in
                [(-w, -h), (w, -h), (w, h), (-w, h)]], pal["c1"], pal)
    tail = rng.choice(["bl", "br", "l", "r"])
    tails = {"bl": [(-w * 0.4, h), (-w * 0.8, h + R * 0.6), (-w * 0.05, h)],
             "br": [(w * 0.4, h), (w * 0.8, h + R * 0.6), (w * 0.05, h)],
             "l": [(-w, -h * 0.2), (-w - R * 0.6, 0), (-w, h * 0.2)],
             "r": [(w, -h * 0.2), (w + R * 0.6, 0), (w, h * 0.2)]}[tail]
    td.polygon([_lpt(x, y, P, lcx, lcy) for (x, y) in tails], fill=pal["c1"] + (255,))
    nch = 1 if rng.random() < 0.6 else 2
    chs = "".join(rng.choice(_LOGO_GLYPHS) for _ in range(nch))
    ink = (245, 245, 245) if sum(pal["c1"]) < 380 else (15, 15, 15)
    _stamp_text(tile, chs, load_font_for(chs, int(h * (0.9 if nch == 1 else 0.7))),
                ink, None, 1, lcx, lcy, rng.uniform(-10, 10) if rng.random() < 0.4 else 0)


_LOGO_KINDS = [
    ("monogram", _lg_monogram), ("duogram", _lg_duogram),
    ("burst", _lg_burst), ("starbadge", _lg_starbadge),
    ("orb", _lg_orb), ("bolt", _lg_bolt),
    ("rings", _lg_rings), ("drop", _lg_drop),
    ("chevron", _lg_chevron), ("shield", _lg_shield),
    ("bubble", _lg_bubble), ("crown", _lg_crown),
    ("spiral", _lg_spiral), ("cube", _lg_cube),
    ("waves", _lg_waves), ("ribbon", _lg_ribbon),
    ("torii", _lg_torii), ("linked", _lg_linked), ("speechmono", _lg_speechmono),
]


# Simple (non-container) logo kinds, safe to nest inside shields / crests /
# rings: bounded marks that never call the device picker, so no recursion.
_LOGO_SIMPLE = {
    "bolt": _lg_bolt, "drop": _lg_drop, "orb": _lg_orb,
    "rings": _lg_rings, "burst": _lg_burst, "starbadge": _lg_starbadge,
    "mono": _lg_monogram, "spiral": _lg_spiral, "waves": _lg_waves,
    "chev": _lg_chevron, "linked": _lg_linked, "torii": _lg_torii,
    "speechmono": _lg_speechmono,
}


def _inner_device(im, dd, cx, cy, R, pal, rng, P, ink, accent):
    """Shared innards for shield / crest / ring containers.

    Rolls mascot-bust / mini-mark / 1-3 letters / chevrons / bars, so two
    containers with the same outline still differ inside. `im`+`dd` are the
    host image and its Draw (logo tile or emblem bar); `P` is the logo xform
    or None on emblems; `ink` must read on the container fill.
    """
    R = max(8, int(R))
    jx, jy = R * rng.uniform(-0.12, 0.12), R * rng.uniform(-0.1, 0.1)
    pick = rng.choices(["letters", "minilogo", "mascot", "chevron", "bars"],
                       weights=[35, 25, 15, 12, 13])[0]
    if pick == "letters":
        nch = rng.randint(1, 3)
        chs = "".join(rng.choice(_LOGO_GLYPHS) for _ in range(nch))
        size = int(R * (1.0 if nch == 1 else 0.72 if nch == 2 else 0.55))
        col = ink if rng.random() < 0.7 else accent
        tilt = rng.uniform(-15, 15) if rng.random() < 0.5 else 0
        _stamp_text(im, chs, load_font_for(chs, size), col, None, 1,
                    cx + jx, cy + jy, tilt)
    elif pick == "minilogo":
        name = rng.choice(sorted(_LOGO_SIMPLE))
        pp = dict(P) if P else _logo_xform(rng)
        sub = dict(pal, shadow=None, glow=None, double=False)
        _LOGO_SIMPLE[name](im, dd, cx + jx, cy + jy, R * rng.uniform(0.55, 0.7),
                           sub, rng, pp)
    elif pick == "mascot":
        draw_mascot(dd, cx + jx, cy + jy, R * rng.uniform(0.5, 0.65), rng, body=False)
    elif pick == "chevron":
        n = rng.randint(1, 3)
        flip = rng.choice([-1, 1])
        vert = rng.random() < 0.7
        th = R * rng.uniform(0.14, 0.24)
        gap = R * rng.uniform(0.3, 0.45)
        span = R * rng.uniform(0.5, 0.7)
        y0 = cy + jy - gap * (n - 1) / 2
        for i in range(n):
            base = [(-span, th), (0, -th), (span, th),
                    (span, th * 2.4), (0, -th + th * 1.4), (-span, th * 2.4)]
            if not vert:
                base = [(-y, x) for (x, y) in base]
            if flip < 0:
                base = [(-x, y) for (x, y) in base]
            dd.polygon([(cx + jx + x, y0 + i * gap + y) for (x, y) in base],
                       fill=accent + (255,))
    else:
        n = rng.randint(2, 4)
        vert_bars = rng.random() < 0.25
        for k in range(n):
            off = (k - (n - 1) / 2) * R * 0.32
            ln = R * rng.uniform(0.35, 0.6)
            th = max(2, R * 0.1)
            if vert_bars:
                dd.rectangle([cx + jx + off - th, cy + jy - ln,
                              cx + jx + off + th, cy + jy + ln],
                             fill=accent + (255,))
            else:
                dd.rectangle([cx + jx - ln, cy + jy + off - th,
                              cx + jx + ln, cy + jy + off + th],
                             fill=accent + (255,))


def draw_logo(im: Image.Image, cx, cy, r, rng: random.Random):
    """Randomized logo mark, v2 (anti-overfit).

    Renders one of 16 parametric families onto a small local tile — each with
    its own palette mode (vivid / mono+accent / pastel / dark / tonal),
    outline treatment (dark / light / colored / double / none + shadow/glow),
    geometry (stretch / tilt / shear / mirror / off-center) and fill
    (solid / nested-gradient / hollow / striped / duo-echo) — plus an optional
    backing container and 0-2 micro details. All from the caller's seeded rng.
    ``im`` is the host RGBA image; the tile is composited at (cx, cy).
    """
    R = max(8, int(r))
    P = _logo_xform(rng)
    pal = _logo_palette(rng)
    S = 4 * R + 28
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    lcx = S / 2 + P["ox"] * R
    lcy = S / 2 + P["oy"] * R
    if pal["glow"] is not None:
        gr = R * 1.45
        td.ellipse([lcx - gr, lcy - gr, lcx + gr, lcy + gr], fill=pal["glow"])
    _logo_backing(tile, td, lcx, lcy, R, pal, rng, P)
    _, fn = rng.choice(_LOGO_KINDS)
    fn(tile, td, lcx, lcy, R, pal, rng, P)
    _logo_micro(tile, td, lcx, lcy, R, pal, rng, P)
    if rng.random() < 0.08:  # local distress: faint dimmed specks, never holes
        n = max(2, int(R * R * 0.004))
        for _ in range(n):
            x = int(rng.uniform(lcx - R, lcx + R))
            y = int(rng.uniform(lcy - R, lcy + R))
            if 0 <= x < S and 0 <= y < S:
                px = tile.getpixel((x, y))
                if px[3] > 40:
                    tile.putpixel((x, y), (px[0], px[1], px[2], max(40, px[3] // 2)))
    if pal["shadow"] is not None:
        dx, dy = pal["shadow"]
        a = tile.getchannel("A")
        sx0, sy0 = max(0, -dx), max(0, -dy)
        sx1, sy1 = min(S, S - dx), min(S, S - dy)
        sh_a = Image.new("L", (S, S), 0)
        sh_a.paste(a.crop((sx0, sy0, sx1, sy1)), (sx0 + dx, sy0 + dy))
        sh_a = sh_a.point(lambda v: int(v * 0.45))
        blk = Image.new("RGBA", (S, S), (10, 10, 10, 255))
        blk.putalpha(sh_a)
        tile = Image.alpha_composite(blk, tile)
    _comp_tile(im, tile, int(cx - S / 2), int(cy - S / 2))


def _star_points(cx, cy, r_out, r_in, n, rot=-90.0):
    import math
    pts = []
    for i in range(2 * n):
        rr = r_out if i % 2 == 0 else r_in
        a = math.radians(rot + i * 180.0 / n)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def gen_pill_logo(rng: random.Random, np_rng) -> Image.Image:
    root = fake_brand(rng).upper()
    suffix = rng.choice(["", f" {rng.randint(1, 99)}", ".NET", ".COM", " PLUS", " HQ",
                         f"{rng.choice(TLDS).upper()}",
                         f" EP{rng.randint(1, 200)}", " ★"])
    brand = (root + suffix).strip()[:26]
    H = rng.randint(40, 160)
    # measure text first so the canvas fits text + optional logo mark
    f = fit_font_size(brand, 420, H - 26)
    tbb = f.getbbox(brand)
    text_w = tbb[2] - tbb[0]
    has_logo = rng.random() < 0.70
    pos = rng.choices(["left", "right", "overlap", "top", "none"],
                      weights=[45, 15, 15, 10, 15])[0]
    if not has_logo:
        pos = "none"
    if pos == "top" and H < 90:
        pos = "left"  # stacked layout needs height
    lr = max(8, int((H - 20) / 2 * rng.uniform(0.85, 1.25))) if pos != "none" else 0
    if pos == "top":
        lr = max(10, int(H * 0.26))
    reserve = (2 * lr + 14) if pos in ("left", "right") else 0
    wide = (text_w + 34 + reserve) > 320
    tagline = (H >= 80 and rng.random() < 0.35) or (wide and rng.random() < 0.75)
    tline = f"{root.lower()}{rng.randint(1, 99)}{rng.choice(TLDS)}" if tagline else ""
    if wide and not tagline and rng.random() < 0.5:
        # wide filled pill with no footer still needs a second text element
        tline = rng.choice(BADGE_TAGLINES)
        tagline = True
    W = text_w + 34 + reserve
    if tagline:
        W = max(W, len(tline) * 8 + 40)
    W = min(max(W, 180), 600)
    im, d = new_canvas(W, H)
    # pill body: 10 treatments, not just the transparent outline ellipse
    style = rng.choices(["outline", "filled", "double", "open", "shadow",
                         "capsule", "taper", "wavy", "notch", "split"],
                        weights=[16, 16, 10, 10, 12,
                                 8, 7, 7, 7, 7])[0]
    bc = (15, 15, 15)
    if style in ("filled", "shadow"):
        bc = rng.choice([(15, 15, 15), (200, 30, 30), (30, 90, 200), (240, 240, 240),
                         (255, 122, 0), (150, 60, 160)])
        ba = rng.randint(140, 200)
        if style == "shadow":
            d.ellipse([7, 8, W - 1, H - 1], fill=(10, 10, 10, 120))
        d.ellipse([4, 4, W - 4, H - 4], fill=bc + (ba,))
    elif style == "double":
        d.ellipse([4, 4, W - 4, H - 4], fill=(0, 0, 0, 0),
                  outline=(255, 255, 255, 200), width=2)
        d.ellipse([10, 10, W - 10, H - 10], fill=(0, 0, 0, 0),
                  outline=(255, 255, 255, 140), width=1)
    elif style == "open":
        a0 = rng.uniform(0, 360)
        d.arc([4, 4, W - 4, H - 4], start=a0, end=a0 + rng.uniform(200, 300),
              fill=(255, 255, 255, 200), width=3)
    elif style == "capsule":
        bc = rng.choice([(15, 15, 15), (200, 30, 30), (30, 90, 200), (240, 240, 240),
                         (255, 122, 0), (150, 60, 160)])
        ba = rng.randint(140, 200)
        d.rounded_rectangle([4, 4, W - 4, H - 4], radius=H // 2 - 4,
                            fill=bc + (ba,), outline=(255, 255, 255, 180), width=2)
    elif style == "taper":
        bc = rng.choice([(15, 15, 15), (200, 30, 30), (30, 90, 200), (240, 240, 240)])
        ba = rng.randint(140, 200)
        _stroke_poly(d, _taper_pts(W, H, rng), bc + (ba,), (255, 255, 255, 180))
    elif style == "wavy":
        bc = rng.choice([(15, 15, 15), (200, 30, 30), (30, 90, 200), (240, 240, 240)])
        ba = rng.randint(120, 180)
        _stroke_poly(d, _wavy_rect_pts(W, H, rng), bc + (ba,), (255, 255, 255, 180))
    elif style == "notch":
        bc = rng.choice([(15, 15, 15), (200, 30, 30), (30, 90, 200), (240, 240, 240)])
        ba = rng.randint(140, 200)
        _stroke_poly(d, _bookmark_pts(W, H, rng), bc + (ba,), (255, 255, 255, 180))
    elif style == "split":
        bc = rng.choice([(15, 15, 15), (200, 30, 30), (30, 90, 200), (240, 240, 240)])
        ba = rng.randint(140, 200)
        d.ellipse([4, 4, W - 4, H - 4], fill=bc + (ba,))
        mid = H // 2
        d.rectangle([6, mid, W - 6, H - 6], fill=(245, 245, 245, 90))
        d.line([(8, mid), (W - 8, mid)], fill=(255, 255, 255, 180), width=1)
    else:
        d.ellipse([4, 4, W - 4, H - 4], fill=(0, 0, 0, 0),
                  outline=(255, 255, 255, rng.randint(150, 220)), width=3)
    # icon placement: left-docked / right / overlapping text edge / stacked
    if pos == "left":
        draw_logo(im, lr + 10, H // 2, lr, rng)
        tx0 = 2 * lr + 20
    elif pos == "overlap":
        draw_logo(im, 14 + lr, H // 2, lr, rng)
        tx0 = 14 + lr // 2  # text starts over the mark's right half
    elif pos == "top":
        draw_logo(im, W // 2, lr + 8, lr, rng)
        tx0 = 0
    else:
        tx0 = 0
    # text coloring: multicolor / single / alternating / gradient-lerp / vgrad
    mode = rng.choices(["multi", "single", "alt", "lerp", "vgrad"],
                       weights=[22, 28, 14, 14, 22])[0]
    base1, base2 = pick_text_color(rng), pick_text_color(rng)
    if style in ("filled", "shadow", "capsule", "taper", "wavy", "notch", "split") \
            and sum(bc) < 380:
        single = rng.choice([(255, 255, 255), (255, 215, 0), base1])
    else:
        single = base1
    fills = []
    for i, ch in enumerate(brand):
        if mode == "multi":
            fills.append(pick_text_color(rng))
        elif mode == "single":
            fills.append(single)
        elif mode == "alt":
            fills.append(base1 if i % 2 == 0 else base2)
        else:
            fills.append(_lerp3(base1, base2, i / max(1, len(brand) - 1)))
    bb = f.getbbox(brand)
    avail = W - tx0 - (8 if pos != "right" else 2 * lr + 20)
    x0 = tx0 + max(0, (avail - (bb[2] - bb[0])) // 2) - bb[0]
    y0 = (H - (bb[3] - bb[1])) // 2 - bb[1]
    if pos == "top":
        y0 = lr * 2 + 12 - bb[1]
    if tagline:
        y0 -= 8
    x = x0
    # word-level FX so gradient angle + shadow offset stay consistent per word:
    # vgrad gets a shared 2-stop vertical gradient, other modes keep their
    # solid per-char colors with a shared drop shadow.
    if mode == "vgrad":
        _vgrad_angle = rng.choice(TEXT_GRAD_ANGLES)
        _word_shadow = roll_text_fx(rng, base1, grad_p=0.0, shadow_p=0.40)
        _vgrad_fx_base = {"do_grad": True, "c1": _norm_rgb(base1),
                          "c2": _norm_rgb(base2), "angle": _vgrad_angle,
                          "do_shadow": _word_shadow.get("do_shadow", False),
                          "dx": _word_shadow.get("dx", 2),
                          "dy": _word_shadow.get("dy", 2),
                          "blur": _word_shadow.get("blur", 0),
                          "opacity": _word_shadow.get("opacity", 0.5)}
    else:
        _word_shadow = roll_text_fx(rng, single, grad_p=0.0, shadow_p=0.35)
        _vgrad_fx_base = None
    for ch, fill in zip(brand, fills):
        cb = f.getbbox(ch)
        sc, sw = outline_for(fill, rng)
        if mode == "single" and style in ("filled", "shadow") and rng.random() < 0.5:
            sw = 0  # flat print on the pill body
        if mode == "vgrad":
            draw_fancy_text(im, (x - cb[0], y0), ch, f, fill, sc if sw else None, sw,
                            fx=dict(_vgrad_fx_base))
        elif _word_shadow.get("do_shadow"):
            draw_fancy_text(im, (x - cb[0], y0), ch, f, fill, sc if sw else None, sw,
                            fx=dict(_word_shadow))
        elif sw:
            d.text((x - cb[0], y0), ch, font=f, fill=fill + (255,),
                   stroke_width=sw, stroke_fill=_flat_stroke(sc) + (255,))
        else:
            d.text((x - cb[0], y0), ch, font=f, fill=fill + (255,))
        x += f.getlength(ch)
    d = ImageDraw.Draw(im)
    if pos == "right":
        draw_logo(im, W - lr - 10, H // 2, lr, rng)
    if tagline:
        f2 = load_font_for(tline, max(8, H // 6))
        tb = f2.getbbox(tline)
        ink = (245, 245, 245, 255) if style in ("outline", "double", "open") or sum(bc) < 380 \
            else (15, 15, 15, 255)
        d.text(((W - (tb[2] - tb[0])) // 2 - tb[0], H - (tb[3] - tb[1]) - 8 - tb[1]),
               tline, font=f2, fill=ink)
    return im


# ---------------------------------------------------------------- scanlator emblem+bar
def _draw_bar_shape(d: ImageDraw.ImageDraw, x0, y0, x1, y1, shape: str, fill):
    """Draw bar body. Returns (rule_x0, rule_x1): the straight top-edge span,
    so the thin rules never overshoot past slanted arrow/chevron edges."""
    if shape == "arrow":
        pt = min((x1 - x0) // 4 + 10, (y1 - y0))
        mid = (y0 + y1) // 2
        d.polygon([(x0, y0), (x1 - pt, y0), (x1, mid), (x1 - pt, y1), (x0, y1)], fill=fill)
        return x0, x1 - pt
    elif shape == "chevron":
        pt = min((x1 - x0) // 5 + 8, (y1 - y0))
        notch = min((x1 - x0) // 6 + 6, (y1 - y0) // 2)
        mid = (y0 + y1) // 2
        d.polygon([(x0 + notch, y0), (x1 - pt, y0), (x1, mid),
                   (x1 - pt, y1), (x0 + notch, y1), (x0, mid)], fill=fill)
        return x0 + notch, x1 - pt
    elif shape == "pill":
        rad = (y1 - y0) // 2
        d.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=fill)
        return x0 + rad, x1 - rad
    elif shape == "dblarrow":
        pt = min((x1 - x0) // 6 + 8, (y1 - y0))
        mid = (y0 + y1) // 2
        d.polygon([(x0 + pt, y0), (x1 - pt, y0), (x1, mid),
                   (x1 - pt, y1), (x0 + pt, y1), (x0, mid)], fill=fill)
        return x0 + pt, x1 - pt
    elif shape == "notch":
        # flat bar with small V notch cut on the right end
        pt = min(14, (y1 - y0) // 2)
        mid = (y0 + y1) // 2
        d.polygon([(x0, y0), (x1, y0), (x1, mid - pt),
                   (x1 - pt, mid), (x1, mid + pt), (x1, y1), (x0, y1)], fill=fill)
        return x0, x1
    elif shape == "taper":
        inset = min((x1 - x0) * 0.06, (y1 - y0) * 0.35)
        d.polygon([(x0, y0), (x1, y0), (x1 - inset, y1), (x0 + inset, y1)], fill=fill)
        return x0 + inset, x1 - inset
    elif shape == "chamfer":
        c = min(14, (y1 - y0) // 2)
        d.polygon([(x0 + c, y0), (x1 - c, y0), (x1, y0 + c), (x1, y1 - c),
                   (x1 - c, y1), (x0 + c, y1), (x0, y1 - c), (x0, y0 + c)], fill=fill)
        return x0 + c, x1 - c
    elif shape == "wavy":
        import math as _m
        n, amp, ph = 24, max(2, (y1 - y0) * 0.08), 1.3
        top = [(x0 + (x1 - x0) * i / n, y0 + amp * _m.sin(i / n * 9.4 + ph)) for i in range(n + 1)]
        bot = [(x1 - (x1 - x0) * i / n, y1 + amp * _m.sin(i / n * 9.4 + ph + 2.1)) for i in range(n + 1)]
        d.polygon(top + bot, fill=fill)
        return x0, x1
    else:  # rect
        d.rectangle([x0, y0, x1, y1], fill=fill)
        return x0, x1


def _draw_emblem(im: Image.Image, d: ImageDraw.ImageDraw, cx, cy, r, kind: str,
                 rng: random.Random, dark_bar: bool):
    """Scanlator emblem mark centered at (cx, cy). 15 structurally distinct
    families, each with internal geometry params (counts / gaps / openings /
    aspects / offsets) so repeats of one kind still differ in silhouette."""
    ring = rng.choice([(235, 40, 40), (255, 122, 0), (150, 90, 255),
                       (255, 255, 255), (255, 215, 0), (40, 200, 140),
                       (50, 140, 255), (255, 90, 180), (60, 220, 130),
                       (240, 240, 240), (20, 20, 20), (255, 170, 60)])
    # disc: mostly dark/light, sometimes brand-colored so emblem pops off the bar
    dr = rng.random()
    if dr < 0.55:
        disc = (12, 12, 12, 255)
    elif dr < 0.75:
        disc = (240, 240, 240, 255)
    elif dr < 0.90:
        disc = rng.choice([(200, 30, 30), (30, 90, 200), (150, 60, 160),
                           (255, 122, 0), (30, 130, 70), (40, 40, 55)]) + (255,)
    else:
        disc = (0, 0, 0, 0)  # no disc: floating mark, no plateau
    ink = (240, 240, 240, 255) if disc[0] < 128 else (15, 15, 15, 255)
    if disc[3] == 0:
        ink = accent_ink = ring + (255,)
    accent = ring + (255,)
    ow = rng.choices([1, 2, 3, 4, 5], weights=[12, 28, 32, 20, 8])[0]
    if kind == "gem":
        # faceted gem: cut / facet-line set / shaded facets / highlight all vary
        cut = rng.choice(["diamond", "hex", "emerald"])
        sx = rng.uniform(0.8, 1.2)
        if cut == "diamond":
            pts = [(cx, cy - r), (cx + r * sx, cy), (cx, cy + r), (cx - r * sx, cy)]
        elif cut == "hex":
            pts = [(cx - r * 0.5 * sx, cy - r), (cx + r * 0.5 * sx, cy - r),
                   (cx + r * sx, cy), (cx + r * 0.5 * sx, cy + r),
                   (cx - r * 0.5 * sx, cy + r), (cx - r * sx, cy)]
        else:
            c = r * 0.35
            pts = [(cx - r * sx + c, cy - r), (cx + r * sx - c, cy - r),
                   (cx + r * sx, cy - r + c), (cx + r * sx, cy + r - c),
                   (cx + r * sx - c, cy + r), (cx - r * sx + c, cy + r),
                   (cx - r * sx, cy + r - c), (cx - r * sx, cy - r + c)]
        d.polygon(pts, fill=disc, outline=accent, width=ow)
        if rng.random() < 0.35:  # inner outline instead of facet lines
            q = [(cx + (x - cx) * 0.68, cy + (y - cy) * 0.68) for (x, y) in pts]
            d.polygon(q, fill=None, outline=accent, width=max(1, ow - 1))
        else:
            lines = rng.sample(["v", "h", "d1", "d2"],
                               rng.randint(1, 3))  # never always the cross
            for L in lines:
                if L == "v":
                    d.line([(cx, cy - r), (cx, cy + r)], fill=accent, width=max(1, ow - 1))
                elif L == "h":
                    d.line([(cx - r * sx, cy), (cx + r * sx, cy)], fill=accent, width=max(1, ow - 1))
                elif L == "d1":
                    d.line([(cx - r * sx, cy - r), (cx + r * sx, cy + r)],
                           fill=accent, width=max(1, ow - 1))
                else:
                    d.line([(cx - r * sx, cy + r), (cx + r * sx, cy - r)],
                           fill=accent, width=max(1, ow - 1))
        for _ in range(rng.randint(0, 2)):  # shaded facet(s), random placement
            qx = rng.choice([-1, 1]) * r * sx * 0.5
            qy = rng.choice([-1, 1]) * r * 0.5
            d.polygon([(cx, cy - r), (cx + qx, cy), (cx + qx * 0.2, cy + qy * 0.2)],
                      fill=ink[:3] + (60,))
        if rng.random() < 0.6:  # highlight in a random corner, or absent
            hx = cx + rng.choice([-1, 1]) * r * sx * 0.45
            hy = cy - r * 0.45 if rng.random() < 0.5 else cy + r * 0.3
            hr = max(2, r * 0.09)
            d.ellipse([hx - hr, hy - hr * 1.4, hx + hr, hy + hr * 1.4],
                      fill=(255, 255, 255, 230))
    elif kind == "horseshoe":
        # thick C arc: opening direction / thickness / studs vary
        opening = rng.choice(["down", "down", "left", "right", "up"])
        spans = {"down": (200, 340), "up": (20, 160),
                 "left": (110, 250), "right": (290, 430)}
        a0, a1 = spans[opening]
        a0 += rng.uniform(-15, 15)
        a1 += rng.uniform(-15, 15)
        disc_style = rng.choice(["filled", "filled", "hollow", "none"])
        if disc_style == "filled":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow)
        elif disc_style == "hollow":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=None, outline=accent, width=ow)
        rr = r * rng.uniform(0.45, 0.6)
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], start=a0, end=a1,
              fill=accent, width=max(3, int(r * rng.uniform(0.18, 0.32))))
        for _ in range(rng.randint(0, 3)):  # studs at random arc positions
            sa = math.radians(rng.uniform(a0, a1))
            px, py = cx + rr * math.cos(sa), cy + rr * math.sin(sa)
            sr = max(2, r * 0.08)
            d.ellipse([px - sr, py - sr, px + sr, py + sr], fill=accent)
    elif kind == "pagoda":
        # stacked temple roofs: count / taper / spire / door / ring all vary
        if rng.random() < 0.7:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow + 1)
            if rng.random() < 0.5:
                d.ellipse([cx - r + 3, cy - r + 3, cx + r - 3, cy + r - 3],
                          fill=(0, 0, 0, 0), outline=accent, width=1)
        n = rng.randint(2, 4)
        w = r * rng.uniform(0.8, 1.0)
        yy = cy + r * 0.45
        for _ in range(n):
            d.polygon([(cx - w, yy), (cx + w, yy), (cx + w * 0.7, yy - r * 0.22),
                        (cx - w * 0.7, yy - r * 0.22)], fill=ink)
            yy -= r * rng.uniform(0.26, 0.34)
            w *= rng.uniform(0.72, 0.84)
        spire = rng.choice(["line", "dot", "crescent", "none"])
        if spire == "line":
            d.line([(cx, yy - r * 0.25), (cx, yy + r * 0.1)], fill=ink, width=2)
        elif spire == "dot":
            sr = max(2, r * 0.08)
            d.ellipse([cx - sr, yy - r * 0.2 - sr, cx + sr, yy - r * 0.2 + sr], fill=ink)
        elif spire == "crescent":
            sr = max(3, r * 0.12)
            d.arc([cx - sr, yy - r * 0.3 - sr, cx + sr, yy - r * 0.3 + sr],
                  start=rng.uniform(0, 360), end=rng.uniform(0, 360) + 270,
                  fill=ink, width=2)
        door = rng.choice(["rect", "arch", "none"])
        if door == "rect":
            d.rectangle([cx - r * 0.18, cy + r * 0.30, cx + r * 0.18, cy + r * 0.48],
                        fill=ink)
        elif door == "arch":
            d.pieslice([cx - r * 0.18, cy + r * 0.18, cx + r * 0.18, cy + r * 0.54],
                       start=180, end=360, fill=ink)
    elif kind == "ring-letter":
        # ring: full / double / broken at random angle; letters: 1-2, offset, tilt
        rs = rng.choice(["full", "double", "broken", "thick"])
        if rs == "full":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow + 1)
        elif rs == "double":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow + 1)
            d.ellipse([cx - r + 5, cy - r + 5, cx + r - 5, cy + r - 5],
                      fill=(0, 0, 0, 0), outline=accent, width=1)
        elif rs == "broken":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc)
            g0 = rng.uniform(0, 360)
            d.arc([cx - r, cy - r, cx + r, cy + r], start=g0 + rng.uniform(25, 90),
                  end=g0 + 360, fill=accent, width=ow + 1)
        else:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent,
                      width=max(4, r // 5))
        center = rng.choices(["letters", "minilogo", "mascot"], weights=[65, 20, 15])[0]
        if center == "letters":
            nch = 1 if rng.random() < 0.7 else 2
            chs = "".join(rng.choice(_LOGO_GLYPHS) for _ in range(nch))
            size = int(r * (1.1 if nch == 1 else 0.8))
            tilt = rng.uniform(-15, 15) if rng.random() < 0.5 else 0
            _stamp_text(im, chs, load_font_for(chs, size), ring, None, 1,
                        cx + r * rng.uniform(-0.12, 0.12), cy + r * rng.uniform(-0.08, 0.08),
                        tilt)
        elif center == "minilogo":
            epal = {"c1": ring, "c2": ink[:3], "back": disc[:3] if disc[3] != 0 else ring,
                    "oc": ink[:3], "ow": 2, "double": False,
                    "shadow": None, "glow": None}
            name = rng.choice(sorted(_LOGO_SIMPLE))
            _LOGO_SIMPLE[name](im, d, cx, cy, r * rng.uniform(0.35, 0.85), epal, rng, _logo_xform(rng))
        else:
            draw_mascot(d, cx, cy, r * rng.uniform(0.40, 0.70), rng, body=False)
    elif kind == "crescent":
        # C-moon: gap size/position vary, round end caps, optional spark
        rr = r * rng.uniform(0.6, 0.75)
        gap = rng.uniform(40, 130)
        g0 = rng.uniform(0, 360)
        wdt = max(2, int(r * rng.uniform(0.18, 0.38)))
        if rng.random() < 0.30:  # floating mark, no disc plateau
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=None,
                      outline=accent, width=max(1, ow - 1))
        else:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow)
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], start=g0 + gap, end=g0 + 360,
              fill=accent, width=wdt)
        for ae in (g0 + gap, g0 + 360):
            sa = math.radians(ae)
            px, py = cx + rr * math.cos(sa), cy + rr * math.sin(sa)
            d.ellipse([px - wdt / 2, py - wdt / 2, px + wdt / 2, py + wdt / 2], fill=accent)
        if rng.random() < 0.45:
            sa = math.radians(g0 + gap / 2)
            rf = rng.uniform(0.25, 0.55)
            sx, sy = cx + rr * rf * math.cos(sa), cy + rr * rf * math.sin(sa)
            sr = max(2, r * rng.uniform(0.07, 0.12))
            if rng.random() < 0.6:
                d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=accent)
            else:
                d.polygon([(sx, sy - sr * 1.6), (sx + sr * 0.4, sy - sr * 0.4),
                           (sx + sr * 1.6, sy), (sx + sr * 0.4, sy + sr * 0.4),
                           (sx, sy + sr * 1.6), (sx - sr * 0.4, sy + sr * 0.4),
                           (sx - sr * 1.6, sy), (sx - sr * 0.4, sy - sr * 0.4)], fill=accent)
    elif kind == "estar":
        # star seal: point count / solid-hollow / center / ring vary
        n = rng.randint(4, 8)
        rot = rng.uniform(0, 360)
        filled = rng.random() < 0.6
        if rng.random() < 0.5:
            if rng.random() < 0.25:  # thick-ring floating variant
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=None,
                          outline=accent, width=max(3, ow + 1))
            else:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow)
        sp = _star_points(cx, cy, r * 0.8, r * 0.36, n, rot)
        if filled:
            d.polygon(sp, fill=accent, outline=accent, width=1)
        else:
            d.polygon(sp, fill=None, outline=accent, width=max(2, ow))
        core = rng.choice(["dot", "dot", "hole", "none", "letter", "ring"])
        jx, jy = r * rng.uniform(-0.1, 0.1), r * rng.uniform(-0.1, 0.1)
        if core == "dot":
            cr = max(2, r * rng.uniform(0.09, 0.15))
            d.ellipse([cx + jx - cr, cy + jy - cr, cx + jx + cr, cy + jy + cr],
                      fill=disc if filled else accent)
        elif core == "hole":
            cr = max(2, r * rng.uniform(0.15, 0.25))
            d.ellipse([cx + jx - cr, cy + jy - cr, cx + jx + cr, cy + jy + cr], fill=disc)
        elif core == "letter":
            ch = rng.choice(_LOGO_GLYPHS)
            _stamp_text(im, ch, load_font_for(ch, int(r * 0.5)),
                        disc if filled else accent, None, 1, cx + jx, cy + jy, 0)
        elif core == "ring":
            cr = max(3, r * 0.2)
            d.ellipse([cx + jx - cr, cy + jy - cr, cx + jx + cr, cy + jy + cr],
                      fill=None, outline=accent, width=2)
    elif kind == "crest":
        # shield crest: proportions vary, innards come from the shared picker
        # (letters / mini-mark / mascot / chevrons / bars)
        w, h = r * rng.uniform(0.75, 0.95), r
        d.polygon([(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy),
                   (cx, cy + h), (cx - w, cy)], fill=disc, outline=accent, width=ow)
        epal = {"c1": ring, "c2": ink[:3], "back": disc[:3] if disc[3] != 0 else ring,
                "oc": ink[:3], "ow": 2, "double": False,
                "shadow": None, "glow": None}
        _inner_device(im, d, cx, cy + r * 0.1, r * rng.uniform(0.35, 0.80), epal, rng, None, ink[:3], ring)
    elif kind == "peaks":
        # mountain mark: peak count/heights vary, sun optional, ring optional
        if rng.random() < 0.6:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow)
        n = rng.randint(2, 4)
        base = cy + r * 0.45
        xs = [cx - r * 0.8 + i * (r * 1.6 / n) for i in range(n + 1)]
        for i in range(n):
            ph = r * rng.uniform(0.5, 1.1)
            x0, x1 = xs[i], xs[i + 1]
            d.polygon([(x0, base), ((x0 + x1) / 2, base - ph), (x1, base)], fill=accent)
        sky = rng.random()
        sx = cx + rng.uniform(-r * 0.4, r * 0.4)
        sy = cy - r * rng.uniform(0.35, 0.6)
        if sky < 0.35:
            sr = max(2, r * rng.uniform(0.10, 0.16))
            d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=accent)
        elif sky < 0.5:
            sr = max(3, r * 0.14)
            d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=None, outline=accent, width=2)
        elif sky < 0.65:
            bw = r * 0.12
            for b in range(rng.randint(1, 2)):
                bx, by = sx + b * r * 0.25, sy + rng.uniform(-r * 0.1, r * 0.1)
                d.line([(bx - bw, by), (bx - bw / 2, by - bw * 0.6), (bx, by),
                        (bx + bw / 2, by - bw * 0.6), (bx + bw, by)],
                       fill=accent, width=2)
    elif kind == "eye":
        # almond eye + offset pupil: gaze breaks mirror symmetry
        w2, h2 = r * rng.uniform(0.85, 1.0), r * rng.uniform(0.42, 0.58)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow)
        d.ellipse([cx - w2, cy - h2, cx + w2, cy + h2], fill=None, outline=accent,
                  width=max(2, ow))
        gaze = rng.uniform(-0.3, 0.3)
        pr = h2 * rng.uniform(0.35, 0.5)
        px = cx + gaze * r
        d.ellipse([px - pr, cy - pr, px + pr, cy + pr], fill=accent)
        if rng.random() < 0.5:
            wr = pr * 0.3
            d.ellipse([px - wr, cy - pr * 0.5 - wr, px - wr + wr * 2, cy - pr * 0.5 + wr],
                      fill=(255, 255, 255, 235))
    elif kind == "yinyang":  # yinyang: two-tone swirl disc, random rotation + dot polarity
        rot = rng.uniform(0, 360)
        cA, cB = disc, accent
        if rng.random() < 0.5:
            cA, cB = cB, cA
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cA, outline=accent, width=ow)
        d.pieslice([cx - r, cy - r, cx + r, cy + r], start=rot, end=rot + 180, fill=cB)
        sa = math.radians(rot + 90 + rng.uniform(-20, 20))
        rr = r * rng.uniform(0.35, 0.6)
        for sgn, col in ((1, cA), (-1, cB)):
            px, py = cx + sgn * rr * math.cos(sa), cy + sgn * rr * math.sin(sa)
            dr = max(2, r * rng.uniform(0.12, 0.2))
            d.ellipse([px - dr, py - dr, px + dr, py + dr], fill=col)
    elif kind == "crown":
        # crown seal: point count / band gems / disc vs floating all vary
        if disc[3] != 0 and rng.random() < 0.7:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow)
        n = rng.randint(3, 5)
        w = r * rng.uniform(0.7, 0.9)
        base_y, tip_y = r * 0.45, -r * rng.uniform(0.6, 0.9)
        raw = [(-w, base_y)]
        for i in range(2 * n - 1):
            x = -w + 2 * w * i / (2 * n - 2)
            y = tip_y * rng.uniform(0.85, 1.1) if i % 2 == 1 else -r * 0.1
            raw.append((x, y))
        raw += [(w, base_y)]
        d.polygon([(cx + x, cy + y) for (x, y) in raw], fill=accent, outline=accent)
        bandh = r * 0.24
        d.polygon([(cx - w, cy + base_y - bandh), (cx + w, cy + base_y - bandh),
                   (cx + w, cy + base_y), (cx - w, cy + base_y)], fill=ink)
        for k in range(rng.randint(1, 3)):
            gx = (k - 1) * w * 0.5 + rng.uniform(-2, 2)
            gr = max(2, r * rng.uniform(0.06, 0.10))
            gy = base_y - bandh / 2
            d.ellipse([cx + gx - gr, cy + gy - gr, cx + gx + gr, cy + gy + gr],
                      fill=disc if disc[3] != 0 else (255, 255, 255, 235))
    elif kind == "bolt":
        # lightning seal: lean / width / duo-echo / hollow all vary
        if disc[3] != 0 and rng.random() < 0.6:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow)
        lean = rng.uniform(-0.35, 0.35)
        w = r * rng.uniform(0.35, 0.6)
        pts = [(cx + lean * r + w * 0.5, cy - r), (cx - w * 0.6, cy + r * 0.15),
               (cx - w * 0.05, cy + r * 0.15), (cx + lean * r - w * 0.5, cy + r),
               (cx + w * 0.6, cy - r * 0.15), (cx + w * 0.05, cy - r * 0.15)]
        if rng.random() < 0.3:
            ex, eyy = rng.uniform(0.15, 0.3) * r, rng.uniform(-0.15, 0.2) * r
            d.polygon([(x + ex, y + eyy) for (x, y) in pts], fill=ink)
        if rng.random() < 0.25:
            d.polygon(pts, fill=None, outline=accent, width=max(2, ow + 1))
        else:
            d.polygon(pts, fill=accent, outline=accent)
    elif kind == "bubble":
        # chat-bubble seal: tail direction / letters / tail length vary
        w, h = r * rng.uniform(0.85, 1.1), r * rng.uniform(0.6, 0.85)
        tail = rng.choice(["bl", "br", "l", "r"])
        cy0 = cy - r * 0.1
        d.rounded_rectangle([cx - w, cy0 - h, cx + w, cy0 + h],
                            radius=max(3, int(r * 0.25)), fill=disc,
                            outline=accent, width=ow)
        tails = {"bl": [(-w * 0.5, h), (-w * 0.9, h + r * 0.7), (-w * 0.1, h)],
                 "br": [(w * 0.5, h), (w * 0.9, h + r * 0.7), (w * 0.1, h)],
                 "l": [(-w, -h * 0.3), (-w - r * 0.7, 0), (-w, h * 0.3)],
                 "r": [(w, -h * 0.3), (w + r * 0.7, 0), (w, h * 0.3)]}[tail]
        d.polygon([(cx + x, cy0 + y) for (x, y) in tails], fill=disc)
        nch = rng.randint(1, 2)
        chs = "".join(rng.choice(_LOGO_GLYPHS) for _ in range(nch))
        _stamp_text(im, chs, load_font_for(chs, int(h * rng.uniform(0.8, 1.0))),
                    ring, None, 1, cx + r * rng.uniform(-0.08, 0.08), cy0, 0)
    elif kind == "bullseye":
        # concentric rings: ring count / solid core / letter core vary
        if disc[3] != 0 and rng.random() < 0.75:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc)
        rings = rng.randint(2, 4)
        scales = [1.0, 0.66, 0.36, 0.18][:rings]
        for t, col in zip(scales, (accent, ink, accent, ink)):
            rr = r * t * rng.uniform(0.90, 1.0)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=None,
                      outline=col, width=max(2, ow))
        core = rng.choice(["dot", "hole", "letter", "none"])
        jx, jy = r * rng.uniform(-0.08, 0.08), r * rng.uniform(-0.08, 0.08)
        if core == "dot":
            cr = max(2, r * rng.uniform(0.10, 0.16))
            d.ellipse([cx + jx - cr, cy + jy - cr, cx + jx + cr, cy + jy + cr], fill=accent)
        elif core == "letter":
            _stamp_text(im, rng.choice(_LOGO_GLYPHS),
                        load_font_for("A", int(r * 0.5)), ring, None, 1,
                        cx + jx, cy + jy, 0)
    else:  # wing: feathered arcs fanning left/right, layered rows vary
        flip = rng.choice([-1, 1])
        if disc[3] != 0 and rng.random() < 0.5:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=disc, outline=accent, width=ow)
        rows = rng.randint(2, 3)
        for row in range(rows):
            yy = cy - r * 0.3 + row * r * 0.45
            ln = r * rng.uniform(0.7, 1.0) - row * r * 0.12
            feathers = rng.randint(3, 5)
            for f in range(feathers):
                t = f / max(1, feathers - 1)
                fx = cx + flip * (-ln + 2 * ln * t)
                fw = r * rng.uniform(0.18, 0.28)
                fh = r * rng.uniform(0.22, 0.34)
                d.ellipse([fx - fw, yy - fh, fx + fw, yy + fh],
                          fill=accent if (f + row) % 2 == 0 else ink)


_EMBLEM_KINDS = ["gem", "ring-letter", "pagoda", "horseshoe", "crescent",
                 "estar", "crest", "peaks", "eye", "yinyang",
                 "crown", "bolt", "bubble", "bullseye", "wing"]


def gen_emblem_bar(rng: random.Random, np_rng) -> Image.Image:
    """Scanlator banner: emblem (circle/diamond) + dark arrow/chevron bar + domain.

    Covers temple/omega/realm/utoon family: dark translucent bar, red/white text,
    ES or EN tagline above the domain, optional mini-chibi on the tail end.
    ~20% of the time emits the bare two-line white text variant (no bar/emblem).
    """
    brand = _procedural_brand(rng) if rng.random() < 0.35 else rng.choice(SCAN_BRANDS)
    num = rng.randint(1, 999) if rng.random() < 0.25 else (rng.randint(1, 99) if rng.random() < 0.5 else "")
    domain = _decorate(f"{brand}{num}{rng.choice(TLDS)}".upper(), rng, p=0.15)
    tag_es = rng.random() < 0.45
    tagline = rng.choice(ES_TAGLINES if tag_es else EN_SCAN_TAGLINES)
    # tagline only sometimes long (temple style has the full-sentence variant)
    if "LINK" in tagline or "CALIDAD" in tagline or "QUALITY" in tagline:
        if rng.random() < 0.6:
            tagline = rng.choice(["LEÉ ANTES EN:", "READ FIRST ON:"])

    # --- bare variant: big outlined domain + small subline, no bar (temple white)
    if rng.random() < 0.2:
        fbig = load_font_for(domain, rng.randint(34, 54))
        bb = fbig.getbbox(domain)
        sub = f"¡{tagline} {domain.lower()}!" if tag_es and len(tagline) < 20 else tagline
        fsmall = load_font_for(sub, rng.randint(13, 18))
        bb2 = fsmall.getbbox(sub)
        W = max(bb[2] - bb[0], bb2[2] - bb2[0]) + rng.randint(20, 36)
        H = (bb[3] - bb[1]) + (bb2[3] - bb2[1]) + rng.randint(18, 26)
        im, d = new_canvas(W, H)
        fill = (255, 255, 255) if rng.random() < 0.7 else (235, 40, 40)
        sc, sw = outline_for(fill, rng)
        draw_fancy_text(im, ((W - (bb[2] - bb[0])) / 2 - bb[0], 4 - bb[1]),
                        domain, fbig, fill, sc, sw, rng=rng)
        y2 = (bb[3] - bb[1]) + 10
        sfill = (255, 255, 255)
        draw_fancy_text(im, ((W - (bb2[2] - bb2[0])) / 2 - bb2[0], y2 - bb2[1]),
                        sub, fsmall, sfill, (10, 10, 10), 2, rng=rng)
        return im

    # --- height mode: SHORT-thin is the point of this pass (user: "short = height")
    hroll = rng.random()
    if hroll < 0.35:
        H = rng.randint(36, 58)  # short/thin strip: single line only
        two_line = False
    elif hroll < 0.75:
        H = rng.randint(70, 130)  # normal
        two_line = rng.random() < 0.55
    else:
        H = rng.randint(130, 170)  # thick block: room for 3 lines
        two_line = True
    three_line = H >= 130 and rng.random() < 0.45
    r = max(10, H // 2 - rng.randint(2, 8))
    kind = rng.choice(_EMBLEM_KINDS)
    shape = rng.choices(["rect", "arrow", "chevron", "pill", "dblarrow", "notch",
                         "taper", "chamfer", "wavy"],
                        weights=[24, 17, 14, 12, 10, 8,
                                 6, 5, 4])[0]
    # thin strips read better as pill/rect; thick blocks as rect/chevron
    if H < 60 and shape in ("chevron", "dblarrow") and rng.random() < 0.6:
        shape = rng.choice(["rect", "pill"])
    dark_bar = rng.random() < 0.8
    bar_rgb = (12, 12, 12) if dark_bar else (232, 232, 232)
    if not dark_bar and rng.random() < 0.3:  # tinted light bars, not just grey
        bar_rgb = rng.choice([(200, 30, 30), (30, 90, 200), (30, 130, 70)])
    bar_alpha = rng.randint(140, 200) if dark_bar else rng.randint(120, 180)
    if H < 60 and rng.random() < 0.4:  # thin strips sometimes faint
        bar_alpha = rng.randint(80, 150)

    # --- emblem scale: larger / smaller logo knob (0.45x tiny .. 1.5x bleed)
    emblem_k = rng.uniform(0.45, 1.50) if rng.random() < 0.80 else 1.0
    er = max(6, int(r * emblem_k))
    # --- emblem side: left (usual) / right mirror / dual (both ends)
    side_roll = rng.random()
    if side_roll < 0.70:
        emblem_side, dual = "left", False
    elif side_roll < 0.85:
        emblem_side, dual = "right", False
    else:
        emblem_side, dual = "left", True
    # bar starts behind emblem vs overlapped on top vs floating gap
    overlap = rng.choices(["behind", "ontop", "gap"], weights=[55, 25, 20])[0]
    if overlap == "behind":
        bar_x0 = r
    elif overlap == "ontop":
        bar_x0 = 0
    else:
        bar_x0 = r + rng.randint(4, 12)
    if emblem_side == "right" and not dual and overlap == "behind":
        bar_x0 = 0  # no left emblem to dock behind: full-width bar, text at left edge

    # --- text measure (font size jitters larger/smaller too)
    dom_size = (H // 2 if not two_line else H // 3 + 6)
    dom_size = int(dom_size * rng.uniform(0.75, 1.30))
    dom_size = max(11, dom_size)
    fdom = load_font_for(domain, dom_size)
    bb = fdom.getbbox(domain)
    dom_w = bb[2] - bb[0]
    tag_w = 0
    ftag = None
    if two_line:
        tag_size = max(10, int(H // 5 * rng.uniform(0.8, 1.25)))
        ftag = load_font_for(tagline, tag_size)
        tag_w = ftag.getbbox(tagline)[2] - ftag.getbbox(tagline)[0]
    subline, fsub, sub_w = "", None, 0
    if three_line:
        subline = f"{brand}{rng.randint(1, 99)}{rng.choice(TLDS)}".lower()
        fsub = load_font_for(subline, max(9, H // 8))
        sub_w = fsub.getbbox(subline)[2] - fsub.getbbox(subline)[0]
    text_w = max(dom_w, tag_w, sub_w)
    # --- tail: mascot head radius scales larger/smaller; tail can be logo instead
    tr_base = max(8, (H - 16) // 3)
    tail_roll = rng.random()
    has_tail = tail_roll < 0.32
    tail_kind = rng.choices(["mascot", "logo", "both"], weights=[60, 25, 15])[0] \
        if has_tail else "none"
    tr = max(6, int(tr_base * rng.uniform(0.6, 1.6))) if has_tail else 0
    tail_reserve = (2 * tr + 18) if tail_kind in ("mascot", "logo") else \
        (4 * tr + 24 if tail_kind == "both" else 0)
    tail_left = has_tail and rng.random() < 0.3
    tail_body = has_tail and rng.random() < 0.25
    # --- bar length: tight/short vs normal vs long banner
    lroll = rng.random()
    if lroll < 0.38:
        pad, w_min = rng.randint(20, 30), 220  # short: hugs text
    elif lroll < 0.78:
        pad, w_min = rng.randint(50, 90), 320  # normal
    else:
        pad, w_min = rng.randint(120, 200), 400  # long banner
    n_emblems = 2 if dual else 1
    W = r * 2 * n_emblems + text_w + pad + tail_reserve
    W = max(W, w_min if H >= 60 else 180)

    im, d = new_canvas(W, H)
    bar_y0, bar_y1 = (4, H - 4) if H < 60 else (6, H - 6)
    bar_x1 = W - 4
    rx0, rx1 = _draw_bar_shape(d, bar_x0, bar_y0, bar_x1, bar_y1, shape,
                               bar_rgb + (bar_alpha,))
    # split two-tone variant: bottom half second tone + divider
    if rng.random() < 0.18:
        c2 = rng.choice([(200, 30, 30), (30, 90, 200), (245, 245, 245),
                         (255, 122, 0), (15, 15, 15)])
        mid = (bar_y0 + bar_y1) // 2
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        _draw_bar_shape(od, bar_x0, mid, bar_x1, bar_y1, "rect", c2 + (90,))
        im.alpha_composite(ov)
        d = ImageDraw.Draw(im)
    # thin top/bottom rules like buzztoon/temple bars (clipped to straight span)
    if rng.random() < 0.85:
        rule_col = (255, 255, 255, 160) if dark_bar else (10, 10, 10, 140)
        d.line([(rx0, bar_y0), (rx1, bar_y0)], fill=rule_col, width=1)
        d.line([(rx0, bar_y1), (rx1, bar_y1)], fill=rule_col, width=1)
    # --- draw emblem(s) with vertical jitter; oversized ones bleed past the bar
    emb_cy = H // 2 + rng.randint(-4, 4)
    if emblem_side == "left" or dual:
        _draw_emblem(im, d, r, emb_cy, er, kind, rng, dark_bar)
    if emblem_side == "right" or dual:
        kind2 = rng.choice(_EMBLEM_KINDS) if dual and rng.random() < 0.6 else kind
        er2 = max(8, int(er * rng.uniform(0.7, 1.0))) if dual else er
        _draw_emblem(im, d, W - r, emb_cy, er2, kind2, rng, dark_bar)

    # domain (+ tagline) inside bar, offset past emblem (or past left tail mascot)
    tx = r * 2 + rng.randint(6, 14)
    if dual:
        tx = r * 2 + rng.randint(6, 14)  # left emblem still docks text
    if emblem_side == "right" and not dual:
        tx = rng.randint(8, 14)  # emblem is on the right: text starts at left edge
    if tail_left:
        tx += tail_reserve
    tail_right_reserve = tail_reserve + 8 if (has_tail and not tail_left) else 14
    if dual or (emblem_side == "right" and not dual):
        tail_right_reserve += 2 * r  # keep text clear of the right emblem
    max_tw = W - tx - tail_right_reserve
    while dom_w > max_tw and fdom.size > 12:
        fdom = load_font_for(domain, fdom.size - 2)
        bb = fdom.getbbox(domain)
        dom_w = bb[2] - bb[0]
    # divider tick between emblem and text, sometimes
    if rng.random() < 0.30 and tx - 8 > bar_x0:
        dx = tx - 5
        d.line([(dx, bar_y0 + 4), (dx, bar_y1 - 4)],
               fill=(255, 255, 255, 120) if dark_bar else (10, 10, 10, 120), width=1)
    stroke_w = rng.choices([0, 1, 2, 3], weights=[12, 28, 45, 15])[0]
    def _stroke_for(fill_rgb):
        if stroke_w == 0:
            return None, 0
        lum = fill_rgb[0] * 0.299 + fill_rgb[1] * 0.587 + fill_rgb[2] * 0.114
        sc = (10, 10, 10, 255) if lum > 130 else (255, 255, 255, 255)
        return sc, stroke_w
    if two_line and ftag is not None:
        tbb = ftag.getbbox(tagline)
        tag_fill = rng.choice([(235, 90, 90), (255, 215, 0), (255, 255, 255)]) \
            if dark_bar else rng.choice([(180, 30, 30), (20, 20, 20)])
        sc0, sw0 = _stroke_for(tag_fill)
        draw_fancy_text(im, (tx - tbb[0], bar_y0 + 4 - tbb[1]), tagline, ftag,
                        tag_fill, sc0, sw0, rng=rng,
                        grad_p=0.15, shadow_p=0.30)
        d = ImageDraw.Draw(im)
        dbb = fdom.getbbox(domain)
        if dark_bar:
            dom_fill = rng.choice([(255, 255, 255), (235, 45, 45), (255, 215, 0),
                                   (120, 220, 255), (200, 170, 255)])
        else:
            dom_fill = rng.choice([(20, 20, 20), (180, 30, 30), (40, 60, 160)])
        sc, sw = _stroke_for(dom_fill)
        dy = bar_y0 + (tbb[3] - tbb[1]) + (6 if H < 100 else 8) - dbb[1]
        if three_line:
            dy = bar_y0 + (tbb[3] - tbb[1]) + 5 - dbb[1]
        draw_fancy_text(im, (tx - dbb[0], dy), domain, fdom, dom_fill, sc, sw,
                        rng=rng)
        d = ImageDraw.Draw(im)
        if three_line and fsub is not None:
            sbb = fsub.getbbox(subline)
            sy = dy + (dbb[3] - dbb[1]) + 3 - sbb[1]
            if sy + (sbb[3] - sbb[1]) < bar_y1 - 2:
                ink3 = (245, 245, 245, 255) if dark_bar else (15, 15, 15, 255)
                d.text((tx - sbb[0], sy), subline, font=fsub, fill=ink3)
    else:
        dbb = fdom.getbbox(domain)
        ty = (H - (dbb[3] - dbb[1])) // 2 - dbb[1]
        if H < 60:
            ty = (bar_y0 + bar_y1 - (dbb[3] - dbb[1])) // 2 - dbb[1]
        if dark_bar:
            tfill = rng.choice([(255, 255, 255), (235, 45, 45), (255, 215, 0),
                                (120, 220, 255), (200, 170, 255)])
        else:
            # light bar -> force dark text so it stays readable
            tfill = rng.choice([(20, 20, 20), (180, 30, 30), (40, 60, 160)])
        sc, sw = _stroke_for(tfill)
        draw_fancy_text(im, (tx - dbb[0], ty), domain, fdom, tfill, sc, sw,
                        rng=rng)
        d = ImageDraw.Draw(im)

    # tail cluster on either end: mascot / logo-mark / both, bust or full body
    if has_tail:
        tail_cy = H // 2 + tr // 3
        if tail_left:
            tail_cx = bar_x0 + tr + 6 if emblem_side != "left" or dual else r * 2 + tr + 6
            if tail_kind in ("mascot", "both"):
                draw_mascot(d, tail_cx, tail_cy, tr, rng, body=tail_body)
            if tail_kind in ("logo", "both"):
                lx = tail_cx + (2 * tr + 8 if tail_kind == "both" else 0)
                draw_logo(im, lx, tail_cy, max(8, int(tr * 0.9)), rng)
                d = ImageDraw.Draw(im)
        else:
            tail_cx = W - tr - 10
            if emblem_side == "right" and not dual:
                tail_cx = W - 2 * r - tr - 10  # sit left of the right emblem
            if tail_kind in ("mascot", "both"):
                draw_mascot(d, tail_cx, tail_cy, tr, rng, body=tail_body)
            if tail_kind in ("logo", "both"):
                lx = tail_cx - (2 * tr + 8 if tail_kind == "both" else 0)
                draw_logo(im, lx, tail_cy, max(8, int(tr * 0.9)), rng)
                d = ImageDraw.Draw(im)
    return im


STYLES = {
    "url_strip": gen_url_strip,
    "stacked_block": gen_stacked_block,
    "number_badge": gen_number_badge,
    "ribbon_bar": gen_ribbon_bar,
    "pill_logo": gen_pill_logo,
    "emblem_bar": gen_emblem_bar,
}
STYLE_NAMES = list(STYLES)


def generate_synthetic_wm(rng: random.Random, np_rng: np.random.Generator,
                          style: str | None = None) -> tuple[Image.Image, str]:
    """Core entry (kept import-ready for later dataset wiring).

    Returns (RGBA watermark, style_name). Deterministic given rng state.
    Degradation stack: global alpha -> edge soften -> speckle -> grain ->
    edge erode/dilate -> directional fade. All rolls come from rng/np_rng.
    """
    style = style or STYLE_NAMES[rng.randrange(len(STYLE_NAMES))]
    im = STYLES[style](rng, np_rng)
    im = apply_global_alpha(im, rng)
    im = soften_alpha(im, rng)
    im = add_speckle(im, np_rng, rng)
    im = add_grain(im, np_rng, rng)
    im = erode_alpha(im, rng)
    im = fade_alpha_edge(im, rng)
    return im, style


def _dhash(im: Image.Image, hash_size: int = 8) -> str:
    """Difference hash of the composited (gray) view + alpha mask.

    Captures visual identity: same text in different colors/sizes still
    hashes differently; exact re-rolls collide. Used for bank dedup.
    """
    g = composite_on_gray(im).convert("L").resize(
        (hash_size + 1, hash_size), Image.BILINEAR)
    px = np.asarray(g, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    a = np.asarray(im.getchannel("A").resize((16, 16), Image.BILINEAR),
                   dtype=np.int64).sum()
    return f"{h:016x}-{int(a % 65536):04x}-{im.width}x{im.height}"


def wm_signature(im: Image.Image, style: str) -> str:
    """Bank dedup key: style + visual dhash + alpha-bucket."""
    a = np.asarray(im.getchannel("A"))
    amax = int(a.max())
    amean = float(a.mean())
    return f"{style}|{_dhash(im)}|amax{amax // 8}|amean{int(amean) // 4}"


# ---------------------------------------------------------------- preview helpers
def composite_on_gray(im: Image.Image, gray: int = 200) -> Image.Image:
    bg = Image.new("RGB", im.size, (gray, gray, gray))
    bg.paste(im, mask=im.getchannel("A"))
    return bg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./synth_wm_preview")
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gray", type=int, default=200)
    ap.add_argument("--style", default=None,
                    help=f"generate only one style (one of: {', '.join(STYLE_NAMES)})")
    ap.add_argument("--bank", action="store_true",
                    help="bank mode: enforce signature-unique outputs, write signatures.jsonl")
    ap.add_argument("--bank-size", type=int, default=0,
                    help="bank target count (defaults to --num when --bank)")
    ap.add_argument("--max-retries", type=int, default=20,
                    help="max re-rolls per slot on signature collision")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "preview").mkdir(parents=True, exist_ok=True)

    # wipe previous run's files: same index can land on a different style, so
    # stale wm_XXXX_<oldstyle>.png / pv_XXXX_<oldstyle>.jpg must not survive
    for stale in list(out.glob("wm_*.png")) + list((out / "preview").glob("pv_*.jpg")):
        try:
            stale.unlink()
        except OSError:
            pass
    for stale in (out / "signatures.jsonl", out / "bank_stats.json"):
        try:
            stale.unlink(missing_ok=True)
        except OSError:
            pass

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    print(f"font probe: {font_used_name(32)}")

    bank_mode = args.bank or args.bank_size > 0
    target = args.bank_size or args.num
    seen: set[str] = set()
    collisions = 0
    sig_f = open(out / "signatures.jsonl", "w", encoding="utf-8") if bank_mode else None

    thumbs: list[Image.Image] = []
    counts: dict[str, int] = {}
    alpha_maxes: list[int] = []
    flat_badges = 0
    i = 0
    attempts = 0
    max_attempts = target * (args.max_retries + 1) + 100
    while i < target and attempts < max_attempts:
        attempts += 1
        wm, style = generate_synthetic_wm(rng, np_rng, style=args.style)
        sig = wm_signature(wm, style)
        if bank_mode and sig in seen:
            collisions += 1
            continue
        seen.add(sig)
        counts[style] = counts.get(style, 0) + 1
        wm.save(out / f"wm_{i:04d}_{style}.png")
        if sig_f is not None:
            import json as _js
            a = np.asarray(wm.getchannel("A"))
            sig_f.write(_js.dumps({"idx": i, "style": style, "sig": sig,
                                   "size": list(wm.size),
                                   "amax": int(a.max()),
                                   "amean": round(float(a.mean()), 1)}) + "\n")
        pv = composite_on_gray(wm, args.gray)
        pv.save(out / "preview" / f"pv_{i:04d}_{style}.jpg", quality=90)
        t = pv.copy()
        t.thumbnail((320, 320))
        thumbs.append(t)
        a = np.asarray(wm.getchannel("A"))
        alpha_maxes.append(int(a.max()))
        # edge-density guardrail: flat empty bodies have low gradient energy
        # under their alpha mask; dense text/graphics raise it. Erode the mask
        # by 1px so the body-vs-background outer edge doesn't inflate the score.
        g = np.asarray(pv.convert("L"), dtype=np.float32)
        m = a > 10
        if m.sum() > 100:
            e = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
            inner = np.zeros_like(m, dtype=bool)
            inner[1:-1, 1:-1] = e
            gx = np.abs(np.diff(g, axis=1))
            gy = np.abs(np.diff(g, axis=0))
            gxm = inner[:, 1:] & inner[:, :-1]
            gym = inner[1:, :] & inner[:-1, :]
            ex = float(gx[gxm].mean()) if gxm.any() else 0.0
            ey = float(gy[gym].mean()) if gym.any() else 0.0
            edge = (ex + ey) / 2
            fx = float(((gx > 20) & gxm).sum() / max(1, gxm.sum())) * 100
            fy = float(((gy > 20) & gym).sum() / max(1, gym.sum())) * 100
            edgefrac = (fx + fy) / 2
        else:
            edge, edgefrac = 0.0, 0.0
        flag = ""
        cov = float((a > 10).mean()) * 100
        # sparse = big SOLID badge/pill plateau with almost no interior edges.
        # Transparent/outline bodies (low cov) are exempt by design. Bars
        # (ribbon/emblem) are legitimately flat — centered text on a wide
        # translucent bar — so only badge + pill bodies are flagged.
        if style in ("number_badge", "pill_logo") \
                and wm.width * wm.height > 35000 and cov > 60 and edgefrac < 6.0:
            flag = " SPARSE?"
            flat_badges += 1
        if i < 60 or (flag and i < 200):
            print(f"wm_{i:04d} {style:13s} {wm.size[0]:3d}x{wm.size[1]:3d} "
                  f"alpha_max={a.max():3d} mean={a.mean():5.1f} edge={edge:5.1f}"
                  f" efrac={edgefrac:4.1f}%{flag}")
        elif i % 500 == 0:
            print(f"... {i}/{target} ...")
        i += 1

    if sig_f is not None:
        sig_f.close()
    # contact sheet (uniform cells sized to tallest thumb + label pad; no overlap).
    # Bank runs cap the sheet at the first 60 thumbs so 10k banks stay viewable.
    sheet_thumbs = thumbs[:60]
    cols = 5
    rows = (len(sheet_thumbs) + cols - 1) // cols
    cw = max(t.width for t in sheet_thumbs) + 20
    chh = max(t.height for t in sheet_thumbs) + 20
    sheet = Image.new("RGB", (cols * cw, rows * chh), (60, 60, 60))
    for idx, t in enumerate(sheet_thumbs):
        x = (idx % cols) * cw + (cw - t.width) // 2
        y = (idx // cols) * chh + (chh - t.height) // 2
        sheet.paste(t, (x, y))
    sheet.save(out / "contact_sheet.jpg", quality=88)
    print(f"\nstyles: {counts}")
    if alpha_maxes:
        import statistics as _st
        print(f"alpha_max: min={min(alpha_maxes)} p50={int(_st.median(alpha_maxes))} "
              f"max={max(alpha_maxes)} mean={sum(alpha_maxes) / len(alpha_maxes):.0f}")
    if bank_mode:
        import json as _js2
        uniq = len(seen)
        (out / "bank_stats.json").write_text(_js2.dumps({
            "target": target, "saved": i, "unique": uniq,
            "collisions": collisions, "attempts": attempts,
            "unique_rate": round(uniq / max(1, attempts), 4),
            "styles": counts, "sparse_flat": flat_badges,
            "seed": args.seed,
        }, indent=2), encoding="utf-8")
        print(f"bank: {uniq}/{target} unique, {collisions} collisions "
              f"({collisions / max(1, attempts) * 100:.1f}% retry rate)")
        if i < target:
            print(f"[warn] bank shortfall: {target - i} missing "
                  f"(raise --max-retries)")
    if flat_badges:
        print(f"[warn] {flat_badges} large solid watermark(s) with efrac<6% (sparse-flat)")
    print(f"saved {i} RGBA + previews -> {out.resolve()}")


if __name__ == "__main__":
    main()

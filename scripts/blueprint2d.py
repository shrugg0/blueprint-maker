#!/usr/bin/env python3
"""2D image (photo, render, drawing, logo) -> technical blueprint sheet.

  blueprint2d.py inspect IMAGE [--out DIR]
  blueprint2d.py build   IMAGE SPEC.json [--out FILE.png]

Pipeline: optional crop -> background mask (alpha / auto corner colour / mask file)
-> line extraction (Canny or XDoG sketch) -> hidden-line style rendering on the blueprint
sheet with outline emphasis, bounding-box dimensions and numbered callouts.
"""
import argparse, json, os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bp_common import Sheet, layout, draw_callouts, format_len, finish, FONT_FILES, UNIT_NAMES

MAXSIDE = 3200


def load_image(path, crop=None):
    im = Image.open(path)
    alpha = None
    if im.mode in ('RGBA', 'LA', 'PA') or 'transparency' in im.info:
        rgba = im.convert('RGBA')
        a = np.array(rgba.split()[-1])
        if a.min() < 250:
            alpha = a
        bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255)); bg.alpha_composite(rgba)
        im = bg.convert('RGB')
    else:
        im = im.convert('RGB')
    rgb = np.array(im)
    off = (0, 0)
    if crop:
        x0, y0, x1, y1 = [int(v) for v in crop]
        rgb = rgb[y0:y1, x0:x1]
        alpha = alpha[y0:y1, x0:x1] if alpha is not None else None
        off = (x0, y0)
    return rgb, alpha, off


def bg_mask(rgb, alpha, mode='auto', mask_path=None, thr=30):
    """Foreground mask (bool HxW) or None."""
    h, w = rgb.shape[:2]
    if mode == 'none':
        return None
    if mask_path:
        m = np.array(Image.open(mask_path).convert('L').resize((w, h))) > 127
        return m
    if alpha is not None:
        return alpha > 128
    k = max(4, min(h, w) // 25)
    corners = np.concatenate([rgb[:k, :k].reshape(-1, 3), rgb[:k, -k:].reshape(-1, 3),
                              rgb[-k:, :k].reshape(-1, 3), rgb[-k:, -k:].reshape(-1, 3)])
    med = np.median(corners, 0)
    if np.abs(corners - med).max(1).mean() > 22:     # corners are not a flat colour -> no reliable bg
        return None
    diff = np.abs(rgb.astype(int) - med[None, None].astype(int)).max(2)
    m = diff > thr
    ks = max(3, (min(h, w) // 150) | 1)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((ks, ks), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((ks, ks), np.uint8)).astype(bool)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s > 0.002 * h * w]
        m = np.isin(lab, keep)
    frac = m.mean()
    return m if 0.03 < frac < 0.92 else None


def extract_lines(rgb, mode='edges', detail=0.5):
    """Binary line map at the resolution of rgb."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = max(gray.shape) / 1400.0
    if mode == 'sketch':
        s = (0.9 + 1.6 * (1 - detail)) * k
        g1 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), s)
        g2 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), s * 1.6)
        d = g1 - 0.98 * g2
        thr = np.percentile(d, 4 + 8 * detail)
        lines = d < min(thr, -1.0)
    else:
        s = (0.8 + 1.8 * (1 - detail)) * k
        g = cv2.GaussianBlur(gray, (0, 0), s)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        hi = float(np.percentile(mag, 97 - 9 * detail))
        hi8 = np.clip(hi / 4.0, 8, 250)   # Sobel(3x3) magnitude ~ 4x gray step
        lines = cv2.Canny(g, hi8 * 0.4, hi8) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(6, int(30 * k))
    return keep[lab]


def to_layer(lines, mask, pal, fill, thick_feat, thick_sil, lum):
    h, w = lines.shape
    layer = np.zeros((h, w, 4), np.uint8)
    if mask is not None and fill > 0:
        lo = np.array(pal['fill_lo']); hi = np.array(pal['fill_hi'])
        col = (lo[None, None] + (hi - lo)[None, None] * lum[..., None]).astype(np.uint8)
        layer[mask, :3] = col[mask]; layer[mask, 3] = int(255 * fill)
    ker = lambda t: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
    feat = cv2.dilate(lines.astype(np.uint8), ker(thick_feat)) > 0 if thick_feat > 0 else lines
    if mask is not None:
        feat &= cv2.dilate(mask.astype(np.uint8), ker(3)) > 0
    layer[feat] = tuple(pal['feat']) + (255,)
    if mask is not None:
        edge = mask & ~(cv2.erode(mask.astype(np.uint8), ker(1)) > 0)
        sil = cv2.dilate(edge.astype(np.uint8), ker(thick_sil)) > 0
        layer[sil] = tuple(pal['sil']) + (255,)
    return layer


# ------------------------------------------------------------------ inspect
def inspect(path, out_dir, spec=None):
    """One contact-sheet image: left = source with a pixel grid (coordinates for "point_px"), right = line preview."""
    os.makedirs(out_dir, exist_ok=True)
    rgb, alpha, off = load_image(path)
    h, w = rgb.shape[:2]
    mask = bg_mask(rgb, alpha)
    print(f'image {w}x{h}px; background mask: {"none (not a flat bg - give a mask or crop)" if mask is None else f"{mask.mean() * 100:.0f}% foreground"}')
    if mask is not None:
        ys, xs = np.nonzero(mask)
        print(f'foreground bbox (x0,y0,x1,y1) = {xs.min()},{ys.min()},{xs.max()},{ys.max()}')
    s = min(1.0, 740 / w, 760 / h)
    sw, sh_ = max(1, int(w * s)), max(1, int(h * s))
    prev = Image.fromarray(rgb).resize((sw, sh_), Image.LANCZOS).convert('RGBA')
    dr = ImageDraw.Draw(prev)
    f = ImageFont.truetype(FONT_FILES['mono'], 14) if FONT_FILES.get('mono') else ImageFont.load_default()
    step = next(v for v in (50, 100, 200, 250, 500, 1000, 2000, 5000) if w / v <= 8)
    for x in range(0, w, step):
        dr.line([x * s, 0, x * s, sh_], fill=(255, 255, 0, 90), width=1); dr.text((x * s + 3, 2), str(x), fill=(255, 255, 0, 255), font=f)
    for y in range(0, h, step):
        dr.line([0, y * s, sw, y * s], fill=(255, 255, 0, 90), width=1); dr.text((3, y * s + 2), str(y), fill=(255, 255, 0, 255), font=f)
    lines = extract_lines(rgb)
    lp = Image.fromarray(((~lines) * 255).astype(np.uint8)).resize((sw, sh_), Image.LANCZOS).convert('RGB')
    sheet = Image.new('RGB', (sw * 2 + 10, sh_), (255, 255, 255))
    sheet.paste(prev.convert('RGB'), (0, 0)); sheet.paste(lp, (sw + 10, 0))
    sheet.save(os.path.join(out_dir, 'inspect.png'))
    print(f'wrote {out_dir}/inspect.png: left = source with grid in SOURCE pixels (use for "point_px": [x, y]); '
          'right = what the line extractor sees (tune "detail" 0.3-0.8, "mode": edges|sketch)')


# ------------------------------------------------------------------ build
def build(path, spec, out):
    rgb, alpha, off = load_image(path, spec.get('crop'))
    mask = bg_mask(rgb, alpha, spec.get('bg', 'auto'), spec.get('mask'))
    pal_name = spec.get('palette', 'blueprint')
    W, H = int(spec.get('width', 1920)), int(spec.get('height', 1080))
    sh = Sheet(W, H, pal_name, spec.get('top_safe', 76))
    pal = sh.pal
    B = {k: sh.box(v) for k, v in layout('single').items()}
    sh.frame()
    mb = B['main']
    sh.panel(mb, spec.get('view_title', 'VIEW A  |  PRIMARY VIEW'), spec.get('view_sub', 'LINE EXTRACTION'))

    # working crop to foreground bbox (+pad) so the subject fills the panel
    if mask is not None and spec.get('fit_subject', True):
        ys, xs = np.nonzero(mask)
        pad = int(0.04 * max(xs.max() - xs.min(), ys.max() - ys.min()))
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(rgb.shape[1], xs.max() + pad), min(rgb.shape[0], ys.max() + pad)
    else:
        x0, y0, x1, y1 = 0, 0, rgb.shape[1], rgb.shape[0]
    sub = rgb[y0:y1, x0:x1]; msub = mask[y0:y1, x0:x1] if mask is not None else None
    inner = (mb[0] + 70, mb[1] + 40, mb[2] - 80, mb[3] - 62)      # ref units, leaves room for dimension lines
    pw_max, ph_max = sh.region_px(inner)
    sc = min(pw_max / sub.shape[1], ph_max / sub.shape[0])
    pw, ph = int(sub.shape[1] * sc), int(sub.shape[0] * sc)
    sub_r = cv2.resize(sub, (pw, ph), interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
    m_r = None
    if msub is not None:
        m_r = cv2.resize(msub.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0
    lines = extract_lines(sub_r, spec.get('mode', 'edges'), spec.get('detail', 0.5))
    lum = cv2.cvtColor(sub_r, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    layer = to_layer(lines, m_r, pal, spec.get('fill', 0.55), 1 if sh.S >= 3 else 0, 2, lum)
    ox, oy = (pw_max - pw) / 2.0, (ph_max - ph) / 2.0              # canvas px offset inside inner
    px_ref = lambda cxp, cyp: (inner[0] + cxp / (sh.fx * sh.S), inner[1] + cyp / (sh.fy * sh.S))
    org = px_ref(ox, oy)
    sh.paste(Image.fromarray(layer, 'RGBA'), org[0], org[1])
    if m_r is None:   # no silhouette available: outline the image area itself
        sh.rect(org[0], org[1], org[0] + pw / (sh.fx * sh.S), org[1] + ph / (sh.fy * sh.S), pal['sil'], 1.2)

    # subject bbox in ref units
    if m_r is not None:
        ys, xs = np.nonzero(m_r)
        bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    else:
        bx0, bx1, by0, by1 = 0, pw - 1, 0, ph - 1
    ra = lambda cxp, cyp: np.array(px_ref(ox + cxp, oy + cyp))
    tl, br = ra(bx0, by0), ra(bx1, by1)
    wsrc, hsrc = (bx1 - bx0) / sc, (by1 - by0) / sc
    sr = spec.get('scale_ref'); f, unit = 1.0, 'px'
    if sr:
        f = float(sr['value']) / (wsrc if sr['axis'] == 'width' else hsrc); unit = sr.get('unit', 'm')
    lab = lambda v: f'{format_len(v * f)} {unit}'
    sh.hdim(tl[0], br[0], br[1] + 24, lab(wsrc), br[1], br[1])
    sh.vdim(br[0] + 28, tl[1], br[1], lab(hsrc), br[0], br[0])

    # callouts (point_px in ORIGINAL image pixels)
    callouts = spec.get('callouts', [])
    anchors = {}
    for i, c in enumerate(callouts, 1):
        sx, sy = c['point_px']
        anchors[i] = ra(((sx - off[0] - x0) * sc), ((sy - off[1] - y0) * sc))
    cm = np.zeros((ph_max, pw_max), bool)
    iy, ix = int(round(oy)), int(round(ox))
    if m_r is not None:
        cm[iy:iy + ph, ix:ix + pw] = m_r[:cm.shape[0] - iy, :cm.shape[1] - ix]
    else:
        cm[iy:iy + ph, ix:ix + pw] = True
    keepout = sh.mask_to_ref(cm, pad=10)
    obst = [(tl[0] - 10, br[1] + 8, br[0] + 10, br[1] + 42), (br[0] + 10, tl[1] - 10, br[0] + 50, br[1] + 10)]
    draw_callouts(sh, callouts, anchors, mb, keepout, (inner[0], inner[1]), obst)

    # text panels
    if spec.get('description'):
        sh.description_panel(B['desc'], spec.get('designation', ''), spec['description'])
    if spec.get('specs'):
        sh.specs_panel(B['specs'], spec['specs'])
    if spec.get('notes'):
        sh.notes_panel(B['notes'], spec['notes'], spec.get('notes_title', 'NOTES'))
    fields = spec.get('fields')
    if fields is None:
        fields = [(k, v) for k, v in (('SPEC NO', spec.get('spec_no')), ('SHEET', spec.get('sheet', '01 / 01')),
                                      ('UNITS', 'PIXELS (px)' if unit == 'px' else UNIT_NAMES.get(unit, unit)),
                                      ('SCALE', spec.get('scale', 'AS SHOWN')), ('CLASS', spec.get('class')),
                                      ('REV', spec.get('rev'))) if v]
    sh.title_block(B['title'], spec.get('title', 'UNTITLED'), spec.get('subtitle', ''), fields)
    finish(sh, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('inspect'); a.add_argument('image'); a.add_argument('--out', default='blueprint_inspect')
    b = sub.add_parser('build'); b.add_argument('image'); b.add_argument('spec'); b.add_argument('--out', default='blueprint.png')
    args = ap.parse_args()
    if args.cmd == 'inspect':
        inspect(args.image, args.out)
    else:
        build(args.image, json.load(open(args.spec)), args.out)


if __name__ == '__main__':
    main()

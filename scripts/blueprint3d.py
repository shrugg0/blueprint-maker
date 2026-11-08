#!/usr/bin/env python3
"""3D model -> technical blueprint wallpaper/sheet.

  blueprint3d.py inspect MODEL [--up +Y --front -X --az 58 --el 20 --k 16] [--out DIR]
  blueprint3d.py build   MODEL SPEC.json [--out FILE.png] [--up ..] [--front ..] [--no-cache]

Canonical frame used everywhere (after --up/--front remap):
  +y = up, the front of the object points toward -x, z = lateral (x cross y).
"""
import argparse, io, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlr
import binding
from hlr import load_mesh, load_raw, Prep, render, render_cached, projection, make_basis, AX
from bp_common import Sheet, layout, draw_callouts, format_len, finish, FONT_FILES, UNIT_NAMES

REF = (1500, 860, 0.03)          # reference render used for "pick" pixels and candidates


def default_up(path):
    return '+Y' if path.lower().endswith(('.glb', '.gltf')) else '+Z'


def cam_vec(az_d, el_d):
    el, az = np.radians(el_d), np.radians(az_d)
    return np.array([-np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])


def model_key(path, up, front, max_faces):
    st = os.stat(path)
    return f'{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{up}|{front}|{max_faces}'


# ------------------------------------------------------------------ anchors / candidates
def ref_proj(V, cam):
    r, u, d = make_basis(-cam, [0, 1, 0])
    X = V @ r; Y = V @ u
    x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
    scr = min(REF[0] * (1 - 2 * REF[2]) / (x1 - x0), REF[1] * (1 - 2 * REF[2]) / (y1 - y0))
    return (X - (x0 + x1) / 2) * scr + REF[0] / 2, REF[1] / 2 - (Y - (y0 + y1) / 2) * scr, V @ d


def visible_idx(PX, PY, DEP, cell=6, tol=0.03):
    key = (PY // cell).astype(np.int64) * 100000 + (PX // cell).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    zmin = np.full(len(uk), np.inf); np.minimum.at(zmin, inv, DEP)
    return np.nonzero(DEP <= zmin[inv] + tol * (DEP.max() - DEP.min()))[0]


def candidates(V, cam, k=16):
    """Deterministic anchor candidates: k-means centres of the visible surface, snapped to vertices,
    numbered left-to-right. Returns list of dict(point, px, py)."""
    PX, PY, DEP = ref_proj(V, cam)
    vis = visible_idx(PX, PY, DEP)
    if len(vis) > 40000:
        vis = vis[::len(vis) // 40000 + 1]
    P = np.stack([PX[vis], PY[vis]], 1)
    C = [P[np.argmin(P[:, 0])]]
    d = np.linalg.norm(P - C[0], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d)); C.append(P[i]); d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    C = np.array(C)
    for _ in range(8):
        lab = ((P[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(len(C)):
            if (lab == j).any():
                C[j] = P[lab == j].mean(0)
    idx = ((P[:, None, :] - C[None]) ** 2).sum(2).argmin(0)
    sel = vis[idx]
    order = np.argsort(PX[sel])
    return [dict(point=V[sel[i]], px=float(PX[sel[i]]), py=float(PY[sel[i]])) for i in order]


# ------------------------------------------------------------------ inspect
def inspect(model, up, front, out_dir, az=58.0, el=20.0, k=16):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    raw = load_raw(model)
    V0 = np.asarray(raw.vertices, float)
    rext = raw.extents
    print(f'raw: {len(raw.faces)} faces; raw extents x,y,z = {np.round(rext, 3).tolist()}')
    upi = AX[up.strip('+-').lower()]
    hor = [i for i in range(3) if i != upi]
    longest = max(hor, key=lambda i: rext[i])
    print(f'assumed up={up}. Longest horizontal axis is raw {"XYZ"[longest]}: the object most likely faces +/-{"XYZ"[longest]} '
          f'(current --front {front}); the front must appear at the LEFT of the side view, flip the sign if it does not.')

    p = V0[np.random.default_rng(0).choice(len(V0), min(40000, len(V0)), replace=False)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.3), dpi=100)
    for a, (i, j, t) in zip(ax, [(0, 1, 'raw X right / Y up'), (0, 2, 'raw X right / Z up'), (2, 1, 'raw Z right / Y up')]):
        a.scatter(p[:, i], p[:, j], s=0.3, c='k'); a.set_aspect('equal'); a.set_title(t, fontsize=10); a.grid(alpha=.3)
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)
    top = Image.open(buf).convert('RGB')

    m = load_mesh(model, up, front, 700000)
    print(f'canonical: length(x)={m.extents[0]:.3f} height(y)={m.extents[1]:.3f} width(z)={m.extents[2]:.3f}')
    cam = cam_vec(az, el)
    prep = Prep(m, 38.0)
    img, proj, _ = render_cached(prep, -cam, [0, 1, 0], REF[0], REF[1], REF[2], pal=None, lw_feat=1, lw_sil=2,
                                 model_key=model_key(model, up, front, 700000))
    bg = Image.new('RGBA', img.size, (10, 38, 85, 255)); bg.alpha_composite(img)
    dr = ImageDraw.Draw(bg)
    fn = ImageFont.truetype(FONT_FILES['monob'], 18) if FONT_FILES.get('monob') else ImageFont.load_default()
    fs = ImageFont.truetype(FONT_FILES['mono'], 15) if FONT_FILES.get('mono') else ImageFont.load_default()
    for x in range(0, REF[0], 100):
        dr.line([x, 0, x, REF[1]], fill=(255, 255, 255, 28), width=1); dr.text((x + 3, 2), str(x), fill=(255, 230, 120, 255), font=fs)
    for y in range(0, REF[1], 100):
        dr.line([0, y, REF[0], y], fill=(255, 255, 255, 28), width=1); dr.text((3, y + 2), str(y), fill=(255, 230, 120, 255), font=fs)
    cands = candidates(m.vertices, cam, k)
    print(f'view az={az:g} el={el:g} (use the same "view" in the spec so candidate numbers match)')
    print('  rel=[x, y, z]: x 0 = front end ... 1 = rear end, y 0 = lowest ... 1 = highest, z 0..1 across the width')
    for n, c in enumerate(cands, 1):
        x, y = c['px'], c['py']
        dr.ellipse([x - 13, y - 13, x + 13, y + 13], fill=(255, 214, 120, 235), outline=(20, 20, 20, 255), width=2)
        dr.text((x, y), str(n), fill=(10, 10, 10, 255), font=fn, anchor='mm')
        rel = (c['point'] - m.bounds[0]) / np.maximum(m.extents, 1e-12)
        print(f'  cand {n:2d}: pick=[{x:.0f}, {y:.0f}]  canonical=[{c["point"][0]:.3g}, {c["point"][1]:.3g}, {c["point"][2]:.3g}]  '
              f'rel=[{rel[0]:.2f}, {rel[1]:.2f}, {rel[2]:.2f}]')
    main = bg.convert('RGB')
    sc = 1200 / REF[0]
    top = top.resize((1200, int(top.height * 1200 / top.width)), Image.LANCZOS)
    main = main.resize((1200, int(main.height * sc)), Image.LANCZOS)
    sheet = Image.new('RGB', (1200, top.height + main.height), (255, 255, 255))
    sheet.paste(top, (0, 0)); sheet.paste(main, (0, top.height))
    sheet.save(os.path.join(out_dir, 'inspect.png'))
    print(f'wrote {out_dir}/inspect.png: top = raw axes (to find up/front), bottom = 3/4 view with numbered anchor candidates '
          f'(grid labels are in {REF[0]}x{REF[1]} reference pixels, usable as "pick")')


# ------------------------------------------------------------------ build
def build(model, spec, out, up=None, front=None, use_cache=True, spec_path=None, relock=False, use_lock=True):
    t_start = time.time()
    up = up or spec.get('up') or default_up(model)
    front = front or spec.get('front') or '-X'
    max_faces = spec.get('max_faces', 700000)
    use_cache = use_cache and spec.get('cache', True)
    mk = model_key(model, up, front, max_faces)
    m = load_mesh(model, up, front, max_faces)
    V = m.vertices
    ext = m.extents
    W, H = int(spec.get('width', 1920)), int(spec.get('height', 1080))
    sh = Sheet(W, H, spec.get('palette', 'blueprint'), spec.get('top_safe', 76))
    pal = sh.pal
    kind = 'multi' if spec.get('layout', 'multi') == 'multi' else 'single'
    B = {k: sh.box(v) for k, v in layout(kind).items()}

    unit, f = 'u', 1.0
    sr = spec.get('scale_ref')
    if sr:
        f = float(sr['value']) / ext[{'length': 0, 'height': 1, 'width': 2}[sr['axis']]]
        unit = sr.get('unit', 'm')
    lab = lambda v: f'{format_len(v * f)} {unit}'

    prep = Prep(m, float(spec.get('feature_deg', 38.0)))
    prep_small = Prep(m, float(spec.get('feature_deg_small', 44.0)), base=prep)
    sh.frame()

    vw = spec.get('view', {})
    az_d, el_d = vw.get('az', 58), vw.get('el', 20)
    cam = cam_vec(az_d, el_d)
    # ---- anchors (resolved before any heavy work so a bad binding fails fast)
    callouts = spec.get('callouts', [])
    PX0 = PY0 = DEP0 = cand_list = None

    def anchor_of(c):
        nonlocal PX0, PY0, DEP0, cand_list
        if 'point' in c:
            return np.array(c['point'], float)
        if 'cand' in c:
            if cand_list is None:
                cand_list = candidates(V, cam, int(spec.get('candidates', 16)))
            if not 1 <= int(c['cand']) <= len(cand_list):
                raise ValueError(f'callout cand {c["cand"]} out of range (1..{len(cand_list)})')
            return cand_list[int(c['cand']) - 1]['point']
        if 'extreme' in c:
            sx = c['extreme']; axn = AX[sx.lstrip('+-')]; sg = -1.0 if sx.startswith('-') else 1.0
            return V[np.argsort(sg * V[:, axn])[-30:]].mean(0)
        if 'pick' not in c:
            raise ValueError(f'callout "{c.get("title", "?")}" needs an anchor: one of cand, pick, extreme, point')
        if PX0 is None:
            PX0, PY0, DEP0 = ref_proj(V, cam)
        px, py = c['pick']
        dd = np.hypot(PX0 - px, PY0 - py)
        sel = np.nonzero(dd < 14)[0]
        if len(sel) == 0:
            sel = np.argsort(dd)[:10]
        sel = sel[np.argsort(DEP0[sel])][:25]
        return V[sel].mean(0)

    pts3d = [anchor_of(c) for c in callouts]
    if use_lock and spec_path and callouts:
        lp = binding.lock_path(spec_path)
        lock = binding.load(lp)
        settings = dict(up=up, front=front, az=az_d, el=el_d, candidates=int(spec.get('candidates', 16)), max_faces=max_faces)
        res = binding.verify(lock, [binding.anchor_def(c) for c in callouts], pts3d,
                             [c.get('title', '') for c in callouts], binding.fingerprint(m), settings, ext)
        if res['errors'] and not relock:
            raise ValueError('\n'.join(res['errors']))
        for w_ in res['warnings']:
            sh.warnings.append('binding: ' + w_)
        if relock or res['write']:
            binding.save(lp, res['new_lock'])
            print(f'{"re-locked" if relock else "locked"} {len(callouts)} anchors to this model: {os.path.basename(lp)}')

    # ---- main 3/4 view
    mb = B['main']
    sh.panel(mb, 'VIEW A  |  3/4 VIEW', f'ORTHOGRAPHIC PROJECTION  |  AZ {az_d:g}\u00b0  EL {el_d:g}\u00b0')
    reg = (mb[0] + 6, mb[1] + 30, mb[2] - 6, mb[3] - 6)
    w, h = sh.region_px(reg)
    img, proj, _ = render_cached(prep, -cam, [0, 1, 0], w, h, vw.get('margin', 0.045), pal=pal, lw_feat=1, lw_sil=2,
                                 model_key=mk, use_cache=use_cache)
    sh.paste(img, reg[0], reg[1])
    Pm = lambda p: sh.to_ref(proj(p), reg)[0]
    keepout = sh.mask_to_ref(np.array(img)[..., 3] > 0, pad=10)
    origin = (reg[0], reg[1])

    # ---- axis triad (drawn after callouts, reserved now)
    o = np.array([mb[2] - 90, mb[3] - 40.0])
    triad_rect = (o[0] - 10, o[1] - 50, o[0] + 64, o[1] + 12)

    anchors = {i: Pm(pts3d[i - 1]) for i in range(1, len(callouts) + 1)}
    draw_callouts(sh, callouts, anchors, mb, keepout, origin, [triad_rect])

    r_, u_, d_ = make_basis(-cam, [0, 1, 0])
    for axv, nm in ((np.array([1., 0, 0]), 'X'), (np.array([0, 1., 0]), 'Y'), (np.array([0, 0, 1.]), 'Z')):
        dv = np.array([axv @ r_, -(axv @ u_)]) * 40
        sh.line(tuple(o), tuple(o + dv), pal['white'], 1.5); sh.arrow(tuple(o + dv), tuple(dv), 8, pal['white'])
        sh.text(tuple(o + dv * 1.35), nm, 'monob', 14, pal['white'], 'mm')
    sh.circle(o, 2.5, pal['white'], fill=pal['white'])

    # ---- orthographic views
    def ortho(box, title, d, up_, mg):
        sh.panel(box, title)
        rg = (box[0] + mg[0], box[1] + mg[1], box[2] - mg[2], box[3] - mg[3])
        w, h = sh.region_px(rg)
        im, pr, _ = render_cached(prep_small, d, up_, w, h, 0.0, pal=pal, lw_feat=1, lw_sil=2, model_key=mk, use_cache=use_cache)
        sh.paste(im, rg[0], rg[1])
        pts = sh.to_ref(pr(V), rg)
        return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()

    if kind == 'multi':
        b = B['side']
        x0_, x1_, y0_, y1_ = ortho(b, 'VIEW B  |  SIDE', (0, 0, -1), [0, 1, 0], (62, 32, 70, 46))
        sh.line((b[0] + 8, y1_), (b[2] - 8, y1_), pal['line'], 0.8, dash=(10, 4, 2, 4))
        sh.hdim(x0_, x1_, y1_ + 22, lab(ext[0]), y1_, y1_); sh.vdim(x1_ + 26, y0_, y1_, lab(ext[1]), x1_, x1_)
        b = B['top']
        x0_, x1_, y0_, y1_ = ortho(b, 'VIEW C  |  TOP', (0, -1, 0), [0, 0, -1], (24, 30, 70, 40))
        sh.hdim(x0_, x1_, y1_ + 20, lab(ext[0]), y1_, y1_); sh.vdim(x1_ + 26, y0_, y1_, lab(ext[2]), x1_, x1_)
        b = B['front']
        x0_, x1_, y0_, y1_ = ortho(b, 'VIEW D  |  FRONT', (1, 0, 0), [0, 1, 0], (40, 34, 60, 46))
        sh.line((b[0] + 8, y1_), (b[2] - 8, y1_), pal['line'], 0.8, dash=(10, 4, 2, 4))
        sh.hdim(x0_, x1_, y1_ + 22, lab(ext[2]), y1_, y1_); sh.vdim(x1_ + 26, y0_, y1_, lab(ext[1]), x1_, x1_)

    # ---- text panels
    if spec.get('notes'):
        sh.notes_panel(B['notes'], spec['notes'], spec.get('notes_title', 'DESIGN REMARKS'))
    if spec.get('description'):
        sh.description_panel(B['desc'], spec.get('designation', ''), spec['description'])
    if spec.get('specs'):
        sh.specs_panel(B['specs'], spec['specs'])
    fields = spec.get('fields')
    if fields is None:
        fields = [(k, v) for k, v in (('SPEC NO', spec.get('spec_no')), ('SHEET', spec.get('sheet', '01 / 01')),
                                      ('UNITS', 'MODEL UNITS (u)' if unit == 'u' else UNIT_NAMES.get(unit, unit)),
                                      ('SCALE', spec.get('scale', 'AS SHOWN')), ('CLASS', spec.get('class')),
                                      ('REV', spec.get('rev'))) if v]
    sh.title_block(B['title'], spec.get('title', 'UNTITLED'), spec.get('subtitle', ''), fields)
    finish(sh, out)
    print(f'extents (length,height,width) = {np.round(ext * f, 3).tolist()} {unit if sr else "model units"} | '
          f'render cache: {hlr.STATS["hit"]} hit, {hlr.STATS["miss"]} painted | {time.time() - t_start:.1f}s')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('inspect'); a.add_argument('model'); a.add_argument('--up'); a.add_argument('--front', default='-X')
    a.add_argument('--out', default='blueprint_inspect'); a.add_argument('--az', type=float, default=58.0)
    a.add_argument('--el', type=float, default=20.0); a.add_argument('--k', type=int, default=16)
    b = sub.add_parser('build'); b.add_argument('model'); b.add_argument('spec'); b.add_argument('--out', default='blueprint.png')
    b.add_argument('--up'); b.add_argument('--front'); b.add_argument('--no-cache', action='store_true')
    b.add_argument('--relock', action='store_true', help='accept the current model/settings and rewrite <spec>.lock.json')
    b.add_argument('--no-lock', action='store_true', help='skip the spec/model binding check (not recommended)')
    args = ap.parse_args()
    try:
        if args.cmd == 'inspect':
            inspect(args.model, args.up or default_up(args.model), args.front, args.out, args.az, args.el, args.k)
        else:
            build(args.model, json.load(open(args.spec)), args.out, args.up, args.front, not args.no_cache,
                  args.spec, args.relock, not args.no_lock)
    except ValueError as e:
        sys.exit(f'error: {e}')


if __name__ == '__main__':
    main()

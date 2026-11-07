"""Shared drawing + layout engine for blueprint sheets.

All coordinates handed to Sheet methods are in *reference units* of a 1920x1080 sheet.
The sheet is rendered at S x supersampling and downsampled, and automatically scaled to
any output resolution (WxH). Layout geometry is defined in 'base' coords (frame top y=76)
and remapped by Sheet.my() so the top safe-zone (e.g. a desktop bar) can be changed.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTES = {
    'blueprint': dict(bg_in=(16, 62, 122), bg_out=(6, 25, 60), grid=(120, 190, 255), grid_a=0.20,
                      line=(140, 220, 255), txt=(205, 236, 255), dim=(255, 214, 120), white=(240, 250, 255),
                      sub=(150, 200, 235), panel=(8, 34, 78), rule=(70, 130, 190),
                      fill_lo=(7, 30, 70), fill_hi=(20, 66, 125), feat=(110, 215, 255), sil=(235, 248, 255)),
    'graphite': dict(bg_in=(38, 42, 48), bg_out=(12, 14, 18), grid=(210, 210, 210), grid_a=0.12,
                     line=(255, 160, 70), txt=(236, 226, 214), dim=(120, 230, 255), white=(255, 246, 236),
                     sub=(196, 164, 132), panel=(18, 20, 24), rule=(130, 96, 62),
                     fill_lo=(16, 18, 22), fill_hi=(56, 48, 44), feat=(255, 150, 60), sil=(255, 236, 212)),
    'paper': dict(bg_in=(247, 249, 252), bg_out=(224, 232, 242), grid=(60, 110, 180), grid_a=0.20,
                  line=(20, 70, 150), txt=(20, 45, 90), dim=(190, 40, 40), white=(10, 30, 70),
                  sub=(70, 100, 150), panel=(255, 255, 255), rule=(120, 150, 200),
                  fill_lo=(232, 238, 248), fill_hi=(200, 216, 240), feat=(20, 80, 170), sil=(10, 30, 80)),
}


def _find_font(names):
    dirs = ['/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/dejavu', '/Library/Fonts',
            'C:/Windows/Fonts']
    try:
        import matplotlib
        dirs.append(os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf'))
    except Exception:
        pass
    for n in names:
        for d in dirs:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return None


FONT_FILES = {
    'mono': _find_font(['DejaVuSansMono.ttf']), 'monob': _find_font(['DejaVuSansMono-Bold.ttf']),
    'cond': _find_font(['DejaVuSansCondensed.ttf', 'DejaVuSans.ttf']),
    'condb': _find_font(['DejaVuSansCondensed-Bold.ttf', 'DejaVuSans-Bold.ttf']),
}


def make_bg(W, H, S, pal):
    """Radial-gradient background with technical grid, built in stripes to save memory."""
    w, h = W * S, H * S
    out = np.empty((h, w, 3), np.uint8)
    a = np.array(pal['bg_in'], np.float32); b = np.array(pal['bg_out'], np.float32)
    gcol = np.array(pal['grid'], np.float32)
    minor, major = 20 * S * W / 1920, 100 * S * W / 1920
    xs = np.arange(w, dtype=np.float32)
    gx_minor = (np.floor(xs / minor) != np.floor((xs - 1) / minor))
    gx_major = (np.floor(xs / major) != np.floor((xs - 1) / major))
    cx, cy = w * 0.45, h * 0.5
    ga = pal['grid_a']
    step = 256
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        r = np.sqrt(((xs[None] - cx) / (w * 0.75)) ** 2 + ((ys - cy) / (h * 0.8)) ** 2)
        r = np.clip(r, 0, 1)[..., None]
        img = a * (1 - r) + b * r
        gy_minor = (np.floor(ys / minor) != np.floor((ys - 1) / minor))[:, 0]
        gy_major = (np.floor(ys / major) != np.floor((ys - 1) / major))[:, 0]
        g = np.zeros((y1 - y0, w), np.float32)
        g[:, gx_minor] += 0.30 * ga; g[gy_minor, :] += 0.30 * ga
        g[:, gx_major] += 0.35 * ga; g[gy_major, :] += 0.35 * ga
        g = np.clip(g, 0, ga)[..., None]
        out[y0:y1] = (img * (1 - g) + gcol * g).astype(np.uint8)
    return Image.fromarray(out, 'RGB').convert('RGBA')


class Sheet:
    def __init__(self, W=1920, H=1080, palette='blueprint', top_safe=76):
        self.W, self.H = int(W), int(H)
        self.S = 3 if self.W <= 2048 else 2
        self.fx, self.fy = self.W / 1920.0, self.H / 1080.0
        self.fm = min(self.fx, self.fy)
        self.pal = PALETTES[palette]
        self.top = float(top_safe)
        self.k = (1064.0 - self.top) / 988.0
        self.warnings = []
        self.im = make_bg(self.W, self.H, self.S, self.pal)
        self.d = ImageDraw.Draw(self.im)
        self._fc = {}

    # ---- coordinate helpers
    def my(self, y):
        return self.top + (y - 76.0) * self.k

    def box(self, b):
        return (b[0], self.my(b[1]), b[2], self.my(b[3]))

    def cx(self, x): return x * self.fx * self.S
    def cy(self, y): return y * self.fy * self.S

    def region_px(self, r):
        return int(round((r[2] - r[0]) * self.fx * self.S)), int(round((r[3] - r[1]) * self.fy * self.S))

    def to_ref(self, pts, region):
        pts = np.atleast_2d(pts).astype(float)
        return np.stack([pts[:, 0] / (self.fx * self.S) + region[0], pts[:, 1] / (self.fy * self.S) + region[1]], 1)

    def paste(self, img, x, y):
        self.im.alpha_composite(img, (int(round(self.cx(x))), int(round(self.cy(y)))))
        self.d = ImageDraw.Draw(self.im)

    def font(self, kind, px):
        key = (kind, px)
        if key not in self._fc:
            f = FONT_FILES.get(kind)
            size = max(6, int(round(px * self.fm * self.S)))
            self._fc[key] = ImageFont.truetype(f, size) if f else ImageFont.load_default()
        return self._fc[key]

    def lw(self, w):
        return max(1, int(round(w * self.fm * self.S)))

    # ---- primitives
    def line(self, p, q, col=None, w=1.0, dash=None):
        col = col or self.pal['line']
        (x0, y0), (x1, y1) = p, q
        if dash is None:
            self.d.line([self.cx(x0), self.cy(y0), self.cx(x1), self.cy(y1)], fill=col, width=self.lw(w))
            return
        L = float(np.hypot(x1 - x0, y1 - y0))
        if L == 0:
            return
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        t, on, i = 0.0, True, 0
        while t < L:
            t2 = min(L, t + dash[i % len(dash)])
            if on:
                self.d.line([self.cx(x0 + ux * t), self.cy(y0 + uy * t), self.cx(x0 + ux * t2), self.cy(y0 + uy * t2)],
                            fill=col, width=self.lw(w))
            t, on, i = t2, not on, i + 1

    def rect(self, x0, y0, x1, y1, col=None, w=1.0, fill=None):
        col = col or self.pal['line']
        if fill is not None:
            ov = Image.new('RGBA', self.im.size, (0, 0, 0, 0))
            ImageDraw.Draw(ov).rectangle([self.cx(x0), self.cy(y0), self.cx(x1), self.cy(y1)], fill=fill)
            self.im.alpha_composite(ov)
            self.d = ImageDraw.Draw(self.im)
        self.d.rectangle([self.cx(x0), self.cy(y0), self.cx(x1), self.cy(y1)], outline=col, width=self.lw(w))

    def circle(self, c, r, outline=None, fill=None, w=1.2):
        outline = outline or self.pal['white']
        self.d.ellipse([self.cx(c[0] - r), self.cy(c[1] - r), self.cx(c[0] + r), self.cy(c[1] + r)],
                       outline=outline, fill=fill, width=self.lw(w))

    def text(self, xy, s, kind='cond', px=15, col=None, anchor='la'):
        self.d.text((self.cx(xy[0]), self.cy(xy[1])), s, font=self.font(kind, px),
                    fill=col or self.pal['txt'], anchor=anchor)

    def tw(self, s, kind='cond', px=15):
        return self.d.textlength(s, font=self.font(kind, px)) / (self.fx * self.S)

    def wrap(self, s, kind, px, maxw):
        lines, cur = [], ''
        for w_ in s.split():
            t = (cur + ' ' + w_).strip()
            if self.tw(t, kind, px) <= maxw:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w_
        if cur:
            lines.append(cur)
        return lines

    def fit(self, s, kind, maxw, maxh, px_max=15, px_min=12, lh=1.3, what='text'):
        """Largest font size (px_max..px_min) at which s fits in maxw x maxh. Returns (px, lines)."""
        lines = []
        for px in range(int(px_max), int(px_min) - 1, -1):
            lines = self.wrap(s, kind, px, maxw)
            if len(lines) * px * lh <= maxh:
                if px < 14:
                    self.warnings.append(f'{what}: shrunk to {px}px (readability limit is ~14px)')
                return px, lines
        self.warnings.append(f'{what}: does NOT fit even at {px_min}px - shorten the text')
        return px_min, lines

    def arrow(self, tip, dirv, size=8, col=None):
        col = col or self.pal['dim']
        dx, dy = dirv
        L = float(np.hypot(dx, dy)); dx /= L; dy /= L
        bx, by = tip[0] - dx * size, tip[1] - dy * size
        nx, ny = -dy, dx
        pts = [(self.cx(tip[0]), self.cy(tip[1])),
               (self.cx(bx + nx * size * .32), self.cy(by + ny * size * .32)),
               (self.cx(bx - nx * size * .32), self.cy(by - ny * size * .32))]
        self.d.polygon(pts, fill=col)

    def rtext(self, center, s, kind='mono', px=14, col=None, angle=90):
        f = self.font(kind, px)
        wpx = int(self.d.textlength(s, font=f)) + 8
        hpx = int(px * self.fm * self.S * 1.6)
        tmp = Image.new('RGBA', (wpx, hpx), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((wpx / 2, hpx / 2), s, font=f, fill=col or self.pal['dim'], anchor='mm')
        tmp = tmp.rotate(angle, expand=True)
        self.im.alpha_composite(tmp, (int(self.cx(center[0]) - tmp.width / 2), int(self.cy(center[1]) - tmp.height / 2)))
        self.d = ImageDraw.Draw(self.im)

    # ---- dimension lines
    def hdim(self, x0, x1, y, text, ext0=None, ext1=None, px=14):
        c = self.pal['dim']
        for x, e in ((x0, ext0), (x1, ext1)):
            if e is not None:
                self.line((x, e), (x, y + (5 if e < y else -5)), c, 0.8, dash=(4, 3))
        tw = self.tw(text, 'mono', px) + 12
        xm = (x0 + x1) / 2
        self.line((x0, y), (xm - tw / 2, y), c, 1.1); self.line((xm + tw / 2, y), (x1, y), c, 1.1)
        self.arrow((x0, y), (-1, 0)); self.arrow((x1, y), (1, 0))
        self.text((xm, y), text, 'mono', px, c, 'mm')

    def vdim(self, x, y0, y1, text, ext0=None, ext1=None, px=14):
        c = self.pal['dim']
        for y, e in ((y0, ext0), (y1, ext1)):
            if e is not None:
                self.line((e, y), (x + (5 if e < x else -5), y), c, 0.8, dash=(4, 3))
        th = self.tw(text, 'mono', px) * (self.fx / self.fy) + 12
        ym = (y0 + y1) / 2
        self.line((x, y0), (x, ym - th / 2), c, 1.1); self.line((x, ym + th / 2), (x, y1), c, 1.1)
        self.arrow((x, y0), (0, -1)); self.arrow((x, y1), (0, 1))
        self.rtext((x, ym), text, 'mono', px, c)

    # ---- frame / panels
    def frame(self):
        P = self.pal
        t = self.top
        self.rect(16, t, 1904, 1064, P['white'], 2.0)
        self.rect(22, t + 6, 1898, 1058, P['line'], 1.0)
        for i in range(1, 12):
            x = 16 + i * (1888 / 12)
            self.line((x, t), (x, t + 6), P['white'], 1); self.line((x, 1058), (x, 1064), P['white'], 1)
        for i in range(1, 6):
            y = t + i * ((1064 - t) / 6)
            self.line((16, y), (22, y), P['white'], 1); self.line((1898, y), (1904, y), P['white'], 1)

    def panel(self, b, title, sub=None):
        P = self.pal
        x0, y0, x1, y1 = b
        self.rect(x0, y0, x1, y1, P['line'], 1.2)
        for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            self.line((cx, cy), (cx + 14 * sx, cy), P['white'], 2); self.line((cx, cy), (cx, cy + 14 * sy), P['white'], 2)
        self.text((x0 + 12, y0 + 8), title, 'condb', 15, P['white'])
        if sub:
            self.text((x1 - 12, y0 + 9), sub, 'cond', 14, P['sub'], 'ra')

    # ---- text panels
    def description_panel(self, b, designation, text):
        self.panel(b, 'TECHNICAL DESCRIPTION')
        P = self.pal
        y = b[1] + 32
        if designation:
            des = designation if self.tw('DESIGNATION: ' + designation, 'monob', 13) < (b[2] - b[0] - 24) else designation
            lines = self.wrap(('DESIGNATION: ' + des).upper(), 'monob', 13, b[2] - b[0] - 24)
            for ln in lines[:2]:
                self.text((b[0] + 12, y), ln, 'monob', 13, P['white']); y += 17
            y += 6
        else:
            y += 4
        px, lines = self.fit(text, 'cond', b[2] - b[0] - 24, b[3] - y - 10, 15, 12, 1.33, 'description')
        for i, ln in enumerate(lines):
            self.text((b[0] + 12, y + i * px * 1.33), ln, 'cond', px, P['txt'])

    def specs_panel(self, b, rows):
        self.panel(b, 'KEY SPECIFICATIONS')
        P = self.pal
        colw = min(150, (b[2] - b[0]) * 0.34)
        vw = b[2] - b[0] - colw - 28
        avail = b[3] - b[1] - 44
        for px in (15, 14, 13, 12):
            wr = [self.wrap(v, 'cond', px, vw) for _, v in rows]
            need = sum(max(1, len(w)) * (px + 3) + 7 for w in wr)
            if need <= avail:
                break
        else:
            self.warnings.append('specs: table does not fit - drop rows or shorten values')
        if px < 14:
            self.warnings.append(f'specs: shrunk to {px}px')
        y = b[1] + 38
        for (k, _), w in zip(rows, wr):
            h = max(1, len(w)) * (px + 3)
            self.text((b[0] + 14, y), k.upper(), 'monob', 13, P['line'])
            for i, ln in enumerate(w):
                self.text((b[0] + 14 + colw, y - 1 + i * (px + 3)), ln, 'cond', px, P['txt'])
            y += h + 3
            self.line((b[0] + 10, y), (b[2] - 10, y), P['rule'], 0.6)
            y += 4

    def notes_panel(self, b, notes, title='DESIGN REMARKS'):
        self.panel(b, title)
        P = self.pal
        tw_ = b[2] - b[0] - 58
        avail = b[3] - b[1] - 48
        for px in (14, 13, 12):
            wr = [self.wrap(n, 'cond', px, tw_) for n in notes]
            need = sum(len(w) * (px + 4) + 12 for w in wr)
            if need <= avail:
                break
        else:
            self.warnings.append('notes: do not fit - drop or shorten notes')
        if px < 14:
            self.warnings.append(f'notes: shrunk to {px}px')
        y = b[1] + 38
        for i, w in enumerate(wr):
            self.circle((b[0] + 22, y + 9), 9, P['line'], w=1.2)
            self.text((b[0] + 22, y + 9), chr(65 + i), 'monob', 13, P['white'], 'mm')
            for j, ln in enumerate(w):
                self.text((b[0] + 40, y + j * (px + 4)), ln, 'cond', px, P['txt'])
            y += len(w) * (px + 4) + 12

    def title_block(self, b, title, subtitle, fields):
        P = self.pal
        self.rect(*b, P['white'], 2.0, fill=P['panel'] + (200,))
        rows = (len(fields) + 1) // 2
        fh = rows * 22
        ty = b[3] - fh - 4
        self.line((b[0], ty), (b[2], ty), P['white'], 1.2)
        mid = b[0] + (b[2] - b[0]) / 2
        self.line((mid, ty), (mid, b[3]), P['white'], 1.2)
        # title + subtitle above the field grid
        avail_h = ty - b[1]
        tpx = 36
        while self.tw(title.upper(), 'condb', tpx) > (b[2] - b[0] - 28) and tpx > 18:
            tpx -= 2
        self.text((b[0] + 14, b[1] + 8), title.upper(), 'condb', tpx, P['white'])
        if subtitle:
            sy = b[1] + 8 + tpx * 1.25
            spx = 15
            while self.tw(subtitle, 'cond', spx) > (b[2] - b[0] - 28) and spx > 12:
                spx -= 1
            self.text((b[0] + 14, sy), subtitle, 'cond', spx, P['txt'])
        for i, (k, v) in enumerate(fields):
            c, r = i % 2, i // 2
            x = b[0] + 12 + c * (mid - b[0]); y = ty + 4 + r * 22
            self.text((x, y), k.upper(), 'monob', 12, P['line'])
            self.text((x + 74, y), str(v).upper(), 'mono', 13, P['white'])
            if c == 0 and r > 0:
                pass
        for r in range(1, rows):
            self.line((b[0], ty + r * 22), (b[2], ty + r * 22), P['rule'], 0.6)

    # ---- callouts
    def label_metrics(self, title, desc, wrap_w=235):
        lines = self.wrap(desc, 'cond', 14, wrap_w)
        tw_title = self.tw(title, 'condb', 14)
        lw_ = [self.tw(l, 'cond', 14) for l in lines]
        bw = max([tw_title] + lw_)
        return dict(lines=lines, line_w=lw_, title_w=tw_title, block_w=bw, w=26 + bw, h=19 + 17 * max(1, len(lines)))

    @staticmethod
    def label_geom(x, y, met, style):
        """Leader start point + bounding rect of a label whose top-left is (x, y)."""
        if style == 'under':
            start = (x + 26 + met['block_w'] + 6, y + 17.0)
        elif style == 'under_legacy':
            start = (x + 26 + met['title_w'] + 6, y + 17.0)
        else:
            start = (x + 10.0, y + 10.0)
        return (x, y, x + met['w'], y + met['h']), start

    def callout(self, n, anchor, title, desc, label, wrap_w=235, style=None):
        """Draw a numbered callout. style: 'tag' (leader leaves the number tag, anchor to the left),
        'under' (leader leaves the right end of the title underline, anchor to the right) or None (legacy rule).
        Returns dict(rect, start, style, met) for auditing."""
        P = self.pal
        lx, ly = label
        met = self.label_metrics(title, desc, wrap_w)
        if style is None:
            style = 'under_legacy' if (wrap_w < 200 or lx < 150) else 'tag'
        rect, start = self.label_geom(lx, ly, met, style)
        tag = (lx + 10, ly + 10)
        if style != 'tag':
            self.line((lx + 26, ly + 17), start, P['white'], 1.0)
        v = np.array(anchor, float) - np.array(start, float)
        v /= (np.linalg.norm(v) or 1.0)
        s0 = np.array(start) + (v * 10 if style == 'tag' else 0)
        self.line(tuple(s0), tuple(np.array(anchor) - v * 6), P['white'], 1.2)
        self.circle(anchor, 6, P['white']); self.circle(anchor, 1.6, P['white'], fill=P['white'])
        self.circle(tag, 10, P['line'], fill=P['panel'] + (255,), w=1.4)
        self.text(tag, str(n), 'monob', 14, P['white'], 'mm')
        self.text((lx + 26, ly), title, 'condb', 14, P['white'])
        for i, ln in enumerate(met['lines']):
            self.text((lx + 26, ly + 19 + i * 17), ln, 'cond', 14, P['txt'])
        text_rects = [(lx + 26, ly, lx + 26 + met['title_w'], ly + 16)] + \
                     [(lx + 26, ly + 19 + i * 17, lx + 26 + met['line_w'][i], ly + 19 + i * 17 + 16) for i in range(len(met['lines']))]
        return dict(rect=rect, text_rects=text_rects, start=start, style=style, met=met, anchor=tuple(anchor), n=n)

    def mask_to_ref(self, mask_canvas, pad=10):
        """Boolean mask in canvas pixels -> keep-out mask in reference units (1 px = 1 ref unit), dilated by pad."""
        from scipy import ndimage as ndi
        h, w = mask_canvas.shape
        nx = int(np.ceil(w / (self.fx * self.S))); ny = int(np.ceil(h / (self.fy * self.S)))
        im = Image.fromarray((mask_canvas.astype(np.uint8)) * 255).resize((nx, ny), Image.BOX)
        return ndi.binary_dilation(np.array(im) > 40, iterations=int(pad))


# ---------------------------------------------------------------- layout
def layout(kind='multi'):
    """Panel boxes in base coords (frame top y=76); pass each through Sheet.box()."""
    if kind == 'multi':
        return dict(main=(40, 96, 1290, 826), side=(1320, 96, 1880, 410), top=(1320, 422, 1880, 600),
                    front=(1320, 612, 1560, 886), notes=(1572, 612, 1880, 886),
                    desc=(40, 838, 700, 1050), specs=(714, 838, 1290, 1050), title=(1320, 898, 1880, 1050))
    return dict(main=(40, 96, 1880, 826), desc=(40, 838, 620, 1050), specs=(634, 838, 1010, 1050),
                notes=(1024, 838, 1306, 1050), title=(1320, 838, 1880, 1050))


# ---------------------------------------------------------------- label placement / audit
def _seg_rect_vec(x0, y0, x1, y1, rx0, ry0, rx1, ry1):
    """Does segment (x0,y0)-(x1,y1) cross rect? All args broadcastable arrays (Liang-Barsky)."""
    x0, y0, x1, y1, rx0, ry0, rx1, ry1 = np.broadcast_arrays(*[np.asarray(v, float) for v in (x0, y0, x1, y1, rx0, ry0, rx1, ry1)])
    dx, dy = x1 - x0, y1 - y0
    t0 = np.zeros(x0.shape); t1 = np.ones(x0.shape); ok = np.ones(x0.shape, bool)
    for p, q in ((-dx, x0 - rx0), (dx, rx1 - x0), (-dy, y0 - ry0), (dy, ry1 - y0)):
        par = p == 0
        ok &= ~(par & (q < 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            r = q / p
        t0 = np.where((p < 0) & ~par, np.maximum(t0, r), t0)
        t1 = np.where((p > 0) & ~par, np.minimum(t1, r), t1)
    return ok & (t0 < t1)


def _seg_seg_vec(ax, ay, bx, by, cx, cy, dx, dy):
    """Proper intersection of segments AB and CD (broadcastable arrays)."""
    cr = lambda ox, oy, px, py, qx, qy: (px - ox) * (qy - oy) - (py - oy) * (qx - ox)
    d1 = cr(ax, ay, bx, by, cx, cy); d2 = cr(ax, ay, bx, by, dx, dy)
    d3 = cr(cx, cy, dx, dy, ax, ay); d4 = cr(cx, cy, dx, dy, bx, by)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


class _Keepout:
    """Summed-area table over a boolean keep-out mask given in reference units."""

    def __init__(self, mask, origin):
        self.ox, self.oy = origin
        if mask is None:
            mask = np.zeros((1, 1), bool)
        self.ny, self.nx = mask.shape
        self.sat = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def area(self, x0, y0, x1, y1):
        xi0 = np.clip(np.floor(np.asarray(x0) - self.ox).astype(int), 0, self.nx)
        xi1 = np.clip(np.ceil(np.asarray(x1) - self.ox).astype(int), 0, self.nx)
        yi0 = np.clip(np.floor(np.asarray(y0) - self.oy).astype(int), 0, self.ny)
        yi1 = np.clip(np.ceil(np.asarray(y1) - self.oy).astype(int), 0, self.ny)
        S = self.sat
        return S[yi1, xi1] - S[yi0, xi1] - S[yi1, xi0] + S[yi0, xi0]


def place_labels(sh, items, anchors, box, keepout=None, origin=(0, 0), obstacles=(), fixed=None, wrap_w=235, step=12):
    """Choose label positions so text avoids the model silhouette and other labels, with short,
    non-crossing leaders. items: {n: (title, desc)} for auto-placed labels; fixed: {n: (rect, start, anchor)} geometry
    of manual labels (kept as given, used as obstacles). Returns {n: (x, y, style)} for the auto ones."""
    ko = _Keepout(keepout, origin)
    fixed = fixed or {}
    placed = {}                       # n -> (rect, start, anchor)
    for n, g in fixed.items():        # manual labels: g = (rect, start, anchor); only used as obstacles
        placed[n] = g
    obst = [tuple(o) for o in obstacles]
    out = {}
    metrics = {n: sh.label_metrics(t, d, wrap_w) for n, (t, d) in items.items()}

    def candidates(n):
        met = metrics[n]; ax, ay = anchors[n]; w, h = met['w'], met['h']
        xs = np.arange(box[0] + 14, max(box[0] + 15, box[2] - 14 - w), step)
        ys = np.arange(box[1] + 34, max(box[1] + 35, box[3] - 14 - h), step)
        X, Y = np.meshgrid(xs, ys); X = X.ravel().astype(float); Y = Y.ravel().astype(float)
        tag = ax <= X + 14
        under = ax >= X + 26 + met['block_w'] + 10
        valid = tag | under
        sx = np.where(tag, X + 10, X + 26 + met['block_w'] + 6); sy = np.where(tag, Y + 10, Y + 17.0)
        rx0, ry0, rx1, ry1 = X, Y, X + w, Y + h
        score = np.hypot(ax - sx, ay - sy)
        score = score + 2.0 * ko.area(rx0, ry0, rx1, ry1)
        gut = np.where(tag, np.abs(X - (box[2] - 14 - w)), np.abs(X - (box[0] + 22)))
        score = score + 1.2 * gut            # prefer tidy columns at the panel sides
        for (r, st, an) in [(p[0], p[1], p[2]) for k, p in placed.items() if k != n]:
            # label rect vs other label rect
            valid &= ~((rx0 < r[2] + 6) & (rx1 > r[0] - 6) & (ry0 < r[3] + 6) & (ry1 > r[1] - 6))
            # my leader vs other label rect
            valid &= ~_seg_rect_vec(sx, sy, ax, ay, r[0] - 2, r[1] - 2, r[2] + 2, r[3] + 2)
            # other leader vs my label rect
            valid &= ~_seg_rect_vec(st[0], st[1], an[0], an[1], rx0 - 2, ry0 - 2, rx1 + 2, ry1 + 2)
            # leader vs leader crossings (soft)
            score = score + 150.0 * _seg_seg_vec(sx, sy, ax, ay, st[0], st[1], an[0], an[1])
        for o in obst:
            valid &= ~((rx0 < o[2] + 6) & (rx1 > o[0] - 6) & (ry0 < o[3] + 6) & (ry1 > o[1] - 6))
            valid &= ~_seg_rect_vec(sx, sy, ax, ay, o[0], o[1], o[2], o[3])
        score = np.where(valid, score, np.inf)
        return X, Y, tag, score, sx, sy, met

    def commit(n):
        X, Y, tag, score, sx, sy, met = candidates(n)
        i = int(np.argmin(score))
        if not np.isfinite(score[i]):
            return False
        x, y = float(X[i]), float(Y[i])
        st = 'tag' if tag[i] else 'under'
        rect, start = sh.label_geom(x, y, met, st)
        placed[n] = (rect, start, tuple(anchors[n]))
        out[n] = (x, y, st)
        return True

    # most constrained first (fewest valid positions)
    order = sorted(items, key=lambda n: int(np.isfinite(candidates(n)[3]).sum()))
    failed = []
    for n in order:
        if not commit(n):
            failed.append(n)
    for _ in range(2):                                  # refinement: re-place each with the others fixed
        for n in list(out):
            old = placed.pop(n); oldo = out.pop(n)
            if not commit(n):
                placed[n] = old; out[n] = oldo
    for n in failed:
        sh.warnings.append(f'layout: no collision-free position found for callout {n}; give it a manual "label"')
        x, y = box[0] + 14, box[1] + 34
        met = metrics[n]; ax = anchors[n][0]
        st = 'tag' if ax <= x + 14 else 'under'
        out[n] = (x, y, st)
    return out


def audit_labels(sh, placed, keepout, origin, box, obstacles=()):
    """Text report on a finished set of callouts. placed: list of dicts returned by Sheet.callout()."""
    ko = _Keepout(keepout, origin)
    W = sh.warnings
    ov = lambda a, b: a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]
    for p in placed:
        n = p['n']; r = p['rect']
        if r[0] < box[0] + 6 or r[1] < box[1] + 24 or r[2] > box[2] - 6 or r[3] > box[3] - 6:
            W.append(f'layout: callout {n} text is outside / too close to the edge of the main panel')
        tot = sum((t[2] - t[0]) * (t[3] - t[1]) for t in p['text_rects'])
        hit = sum(float(ko.area(*t)) for t in p['text_rects'])
        if hit / max(1.0, tot) > 0.03:
            W.append(f'layout: callout {n} text overlaps the model ({hit / max(1.0, tot) * 100:.0f}%)')
        if any(ov(t, o) for t in p['text_rects'] for o in obstacles):
            W.append(f'layout: callout {n} text overlaps the axis triad')
        sx, sy = p['start']; ax, ay = p['anchor']
        for t in p['text_rects'][1:]:      # own description lines (the title is underlined, not crossed)
            if bool(_seg_rect_vec(sx, sy, ax, ay, t[0] + 1, t[1] + 1, t[2] - 1, t[3] - 1)):
                W.append(f'layout: callout {n} leader line crosses its own text')
                break
    for i, p in enumerate(placed):
        for j, q in enumerate(placed):
            if i == j:
                continue
            if i < j and any(ov(a, b) for a in p['text_rects'] for b in q['text_rects']):
                W.append(f"layout: callouts {p['n']} and {q['n']} overlap")
            if any(bool(_seg_rect_vec(*p['start'], *p['anchor'], *t)) for t in q['text_rects']):
                W.append(f"layout: leader of callout {p['n']} crosses the text of callout {q['n']}")
            if i < j and bool(_seg_seg_vec(*p['start'], *p['anchor'], *q['start'], *q['anchor'])):
                W.append(f"layout: leaders of callouts {p['n']} and {q['n']} cross")


def draw_callouts(sh, callouts, anchors, box, keepout, origin, obstacles=()):
    """Place (auto or manual), draw and audit all callouts. callouts: spec dicts with title/desc and optional
    'label': [x, y] (manual, 1920x1080 reference coords) / 'style'. anchors: {n: (x, y)} in reference units."""
    fixed, manual, auto_items = {}, {}, {}
    for i, c in enumerate(callouts, 1):
        if 'label' in c:
            lx, ly = c['label'][0], sh.my(c['label'][1])
            ww = 175 if lx < 150 else 235
            style = c.get('style')
            st = style or ('under_legacy' if (ww < 200 or lx < 150) else 'tag')
            met = sh.label_metrics(c['title'].upper(), c.get('desc', ''), ww)
            rect, start = sh.label_geom(lx, ly, met, st)
            fixed[i] = (rect, start, tuple(anchors[i]))
            manual[i] = (lx, ly, style, ww)
        else:
            auto_items[i] = (c['title'].upper(), c.get('desc', ''))
    auto_pos = place_labels(sh, auto_items, anchors, box, keepout, origin, obstacles, fixed) if auto_items else {}
    placed = []
    for i, c in enumerate(callouts, 1):
        if i in manual:
            lx, ly, style, ww = manual[i]
            placed.append(sh.callout(i, anchors[i], c['title'].upper(), c.get('desc', ''), (lx, ly), ww, style))
        else:
            x, y, st = auto_pos[i]
            placed.append(sh.callout(i, anchors[i], c['title'].upper(), c.get('desc', ''), (x, y), 235, st))
    audit_labels(sh, placed, keepout, origin, box, obstacles)
    return placed


def format_len(v):
    if v >= 100: return f'{v:.0f}'
    if v >= 10: return f'{v:.1f}'
    return f'{v:.2f}'


def finish(sheet, out_path):
    img = sheet.im.convert('RGB')
    if img.size != (sheet.W, sheet.H):
        img = img.resize((sheet.W, sheet.H), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    for w in sheet.warnings:
        print('WARNING:', w)
    print('saved', out_path, img.size)

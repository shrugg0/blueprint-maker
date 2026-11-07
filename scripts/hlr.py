"""Hidden-line renderer for triangle meshes (painter's algorithm + feature/silhouette edges)."""
import numpy as np
import trimesh
from PIL import Image, ImageDraw

AX = {'x': 0, 'y': 1, 'z': 2}


def axis_vec(s):
    s = s.strip().lower()
    sign = -1.0 if s.startswith('-') else 1.0
    v = np.zeros(3)
    v[AX[s.lstrip('+-')]] = sign
    return v


def canon_matrix(up, front):
    """Columns map model axes -> canonical frame: +y up, front of the object toward -x, z = x cross y."""
    yv = axis_vec(up)
    xv = -axis_vec(front)
    if abs(float(np.dot(xv, yv))) > 1e-6:
        raise ValueError('up and front axes must be different axes (e.g. up=+Y front=-X)')
    zv = np.cross(xv, yv)
    return np.stack([xv, yv, zv], 1)


def load_raw(path):
    s = trimesh.load(path, process=False)
    if isinstance(s, trimesh.Scene):
        m = s.to_geometry() if hasattr(s, 'to_geometry') else s.dump(concatenate=True)
    else:
        m = s
    return m


def load_mesh(path, up='+Y', front='-X', max_faces=None):
    m = load_raw(path)
    V = np.asarray(m.vertices, float) @ canon_matrix(up, front)
    F = np.asarray(m.faces).copy()
    # a mirrored basis would flip winding; canon_matrix is right-handed (det +1) by construction
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if max_faces and len(m.faces) > max_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=int(max_faces))
        except Exception as e:  # needs fast_simplification
            print('WARNING: could not decimate (pip install fast-simplification):', e)
    # merge by position so edge adjacency works across UV/normal seams
    span = float(np.max(m.extents)) or 1.0
    digits = max(0, int(2 - np.floor(np.log10(span / 1000.0))))
    m.merge_vertices(digits_vertex=min(digits, 6))
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    return m


class Prep:
    """Edge data for a mesh. Adjacency is computed lazily (only when something is actually painted),
    and can be shared between thresholds via base=."""

    def __init__(self, m, feature_deg=38.0, base=None):
        self.m = m
        self.feature_deg = float(feature_deg)
        self.base = base
        self._ok = False

    def _ensure(self):
        if self._ok:
            return
        if self.base is not None:
            self.base._ensure()
            self._adj, self._edges, self._ang = self.base._adj, self.base._edges, self.base._ang
        else:
            m = self.m
            self._adj = m.face_adjacency
            self._edges = m.face_adjacency_edges
            self._ang = np.degrees(m.face_adjacency_angles)
        self._feature = self._ang > self.feature_deg
        self._ok = True

    adj = property(lambda self: (self._ensure(), self._adj)[1])
    adj_edges = property(lambda self: (self._ensure(), self._edges)[1])
    adj_ang = property(lambda self: (self._ensure(), self._ang)[1])
    feature = property(lambda self: (self._ensure(), self._feature)[1])


def make_basis(d, up):
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    r = np.cross(d, up); r /= np.linalg.norm(r)
    u = np.cross(r, d)
    return r, u, d


def projection(m, d, up, w, h, margin=0.0):
    """Cheap (numpy only) orthographic projection of the mesh into a w x h pixel box.
    Returns (project(world)->px, scale, (r, u, dd))."""
    r, u, dd = make_basis(d, up)
    V = m.vertices
    X = V @ r; Y = V @ u
    x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
    sc = min(w * (1 - 2 * margin) / (x1 - x0), h * (1 - 2 * margin) / (y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def project(p):
        p = np.atleast_2d(p)
        return np.stack([((p @ r) - cx) * sc + w / 2, h / 2 - ((p @ u) - cy) * sc], 1)
    return project, sc, (r, u, dd)


def render(prep, d, up, w, h, margin=0.0, sil_deg=6.0, pal=None, lw_feat=1, lw_sil=2,
           light=(-0.4, 0.8, 0.45)):
    """Orthographic hidden-line render -> (RGBA image, project(world)->px, scale)."""
    pal = pal or {}
    fill_lo = pal.get('fill_lo', (7, 30, 70)); fill_hi = pal.get('fill_hi', (20, 66, 125))
    feat_col = pal.get('feat', (110, 215, 255)); sil_col = pal.get('sil', (235, 248, 255))
    m = prep.m
    project, sc, (r, u, dd) = projection(m, d, up, w, h, margin)
    V = m.vertices
    Z = V @ dd
    P0 = project(V)
    PX, PY = P0[:, 0], P0[:, 1]

    F = m.faces
    fd = Z[F].mean(1)
    order = np.argsort(-fd)
    rank = np.empty(len(F), int); rank[order] = np.arange(len(F))
    L = np.array(light); L = L[0] * r + L[1] * u - L[2] * dd; L /= np.linalg.norm(L)
    nrm = m.face_normals
    lam = np.abs(nrm @ L)
    lo = np.array(fill_lo); hi = np.array(fill_hi)
    cols = (lo[None] + (hi - lo)[None] * lam[:, None]).astype(int)
    qcols = [tuple(c) + (255,) for c in cols.tolist()]

    a, b = prep.adj[:, 0], prep.adj[:, 1]
    facing = nrm @ dd
    sil = (np.sign(facing[a]) != np.sign(facing[b])) & (prep.adj_ang > sil_deg)
    feat = prep.feature & ~sil
    owner_rank = rank[np.where(rank[a] > rank[b], a, b)]

    def bucket(mask):
        idx = np.nonzero(mask)[0]
        o = owner_rank[idx]
        s = np.argsort(o)
        return idx[s].tolist(), o[s].tolist()

    fi, fo = bucket(feat)
    si, so = bucket(sil)
    ev = prep.adj_edges.tolist()
    Pl = np.stack([PX, PY], 1).tolist()
    Fl = F.tolist()
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    fp = sp = 0
    nf, ns = len(fi), len(si)
    fc = tuple(feat_col) + (255,); sc_ = tuple(sil_col) + (255,)
    for k, f in enumerate(order.tolist()):
        t = Fl[f]
        dr.polygon([tuple(Pl[t[0]]), tuple(Pl[t[1]]), tuple(Pl[t[2]])], fill=qcols[f])
        while fp < nf and fo[fp] == k:
            e = ev[fi[fp]]
            dr.line([tuple(Pl[e[0]]), tuple(Pl[e[1]])], fill=fc, width=lw_feat); fp += 1
        while sp < ns and so[sp] == k:
            e = ev[si[sp]]
            dr.line([tuple(Pl[e[0]]), tuple(Pl[e[1]])], fill=sc_, width=lw_sil); sp += 1
    return img, project, sc


# ---------------------------------------------------------------- render cache
import hashlib, json, os, glob

_CODE = hashlib.sha1(open(__file__, 'rb').read()).hexdigest()[:8]
STATS = {'hit': 0, 'miss': 0}


def cache_dir():
    d = os.environ.get('BP_CACHE') or os.path.join(os.path.expanduser('~'), '.cache', 'blueprint-maker')
    os.makedirs(d, exist_ok=True)
    return d


def render_cached(prep, d, up, w, h, margin=0.0, pal=None, lw_feat=1, lw_sil=2, model_key='', use_cache=True):
    """Same as render(), but the painted RGBA layer is cached on disk (keyed by model + view + style)."""
    project, sc, _ = projection(prep.m, d, up, w, h, margin)
    if not use_cache or not model_key:
        return render(prep, d, up, w, h, margin, pal=pal, lw_feat=lw_feat, lw_sil=lw_sil)[0], project, sc
    key = hashlib.sha1(json.dumps([model_key, np.round(np.asarray(d, float), 6).tolist(), list(map(float, up)), w, h,
                                   margin, pal and {k: list(v) for k, v in pal.items() if k in ('fill_lo', 'fill_hi', 'feat', 'sil')},
                                   lw_feat, lw_sil, prep.feature_deg, _CODE], sort_keys=True, default=str).encode()).hexdigest()[:20]
    path = os.path.join(cache_dir(), key + '.png')
    if os.path.exists(path):
        try:
            img = Image.open(path).convert('RGBA'); img.load()
            os.utime(path, None)
            STATS['hit'] += 1
            return img, project, sc
        except Exception:
            pass
    STATS['miss'] += 1
    img = render(prep, d, up, w, h, margin, pal=pal, lw_feat=lw_feat, lw_sil=lw_sil)[0]
    try:
        img.save(path, compress_level=1)
        files = sorted(glob.glob(os.path.join(cache_dir(), '*.png')), key=os.path.getmtime)
        for f in files[:-40]:
            os.remove(f)
    except Exception:
        pass
    return img, project, sc

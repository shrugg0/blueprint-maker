"""Spec <-> model binding (lockfile).

Anchors written as `cand`, `pick` or `extreme` only mean something for the model, orientation and view they were
chosen on. The first build writes <spec>.lock.json with the 3D point every anchor resolved to plus a geometric
fingerprint of the model and the settings that influence anchors. Later builds re-resolve the anchors and compare:

  * anchors drifted  -> hard error (re-inspect, re-pick, then build with --relock)
  * model differs but anchors still land on the same place (e.g. lossless re-export) -> warning
  * spec anchor edited on purpose (same model/settings) -> lock entry is updated silently
"""
import hashlib, json, os
import numpy as np

LOCK_VERSION = 1
ANCHOR_KEYS = ('cand', 'pick', 'extreme', 'point')


def lock_path(spec_path):
    return os.path.splitext(spec_path)[0] + '.lock.json'


def fingerprint(m):
    """Geometry hash independent of vertex order / file format: quantised vertex set + face count."""
    V = np.asarray(m.vertices)
    span = float(np.max(m.extents)) or 1.0
    q = np.round(V / (span * 1e-4)).astype(np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    h = hashlib.sha1()
    h.update(q[order].tobytes()); h.update(str(len(m.faces)).encode())
    return h.hexdigest()[:16]


def anchor_def(c):
    return {k: c[k] for k in ANCHOR_KEYS if k in c}


def load(path):
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
        return d if d.get('version') == LOCK_VERSION else None
    except Exception:
        return None


def save(path, lock):
    json.dump(lock, open(path, 'w'), indent=1)


def verify(lock, defs, pts, titles, fp, settings, extents, tol=0.03):
    """Compare freshly resolved anchors with the lock. Returns dict(errors, warnings, new_lock, write)."""
    diag = float(np.linalg.norm(extents)) or 1.0
    new_lock = dict(version=LOCK_VERSION, model_fingerprint=fp, settings=settings,
                    extents=[round(float(e), 4) for e in extents],
                    callouts=[dict(title=t, anchor_def=d, point=[round(float(v), 4) for v in p])
                              for t, d, p in zip(titles, defs, pts)])
    out = dict(errors=[], warnings=[], new_lock=new_lock, write=lock is None)
    if lock is None:
        return out
    old = lock.get('callouts', [])
    drift = []
    edited = len(old) != len(defs)
    for i, (d, p) in enumerate(zip(defs, pts)):
        if i < len(old) and old[i].get('anchor_def') == d:
            dist = float(np.linalg.norm(np.array(old[i]['point']) - p))
            if dist > tol * diag:
                drift.append((i + 1, titles[i], d, dist))
        else:
            edited = True
    fp_changed = lock.get('model_fingerprint') != fp
    old_s = lock.get('settings', {})
    diffs = [f'{k} {old_s.get(k)} -> {settings.get(k)}' for k in settings if old_s.get(k) != settings.get(k)]
    if drift:
        causes = []
        if fp_changed:
            causes.append('the model geometry is different from the one the spec was locked to')
        if diffs:
            causes.append('anchor-relevant settings changed (' + '; '.join(diffs) + ')')
        if not causes:
            causes.append('anchors resolve differently although model and settings look unchanged')
        lines = [f'spec is not bound to this model: {len(drift)} of {len(defs)} callout anchors moved, so the labels would '
                 f'point at different parts. Cause: ' + ' and '.join(causes) + '.']
        for n, t, d, dist in drift[:8]:
            lines.append(f'  callout {n} "{t}" {json.dumps(d)}: moved {dist:.3g} model units ({dist / diag * 100:.0f}% of the model size)')
        lines.append('Fix: re-run `inspect` with the same settings as the spec, re-pick the anchors, then build with --relock '
                     '(or restore the original model / settings).')
        out['errors'] = lines
        return out
    if fp_changed:
        out['warnings'].append('model differs from the one the spec was locked to, but all anchors still resolve to the same '
                               'place (re-export?). Check the result, then run with --relock to update the lock.')
    elif diffs:
        out['warnings'].append('anchor-relevant settings changed (' + '; '.join(diffs) + ') but anchors did not move.')
    out['write'] = edited and not fp_changed
    return out

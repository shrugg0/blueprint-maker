#!/usr/bin/env python3
"""Spec <-> model binding tests. Run:  python3 tests/test_binding.py

Checks that a saved spec either stays attached to the intended features or fails explicitly when the model,
its orientation or the view changes, and that the escape hatches (--relock, --no-lock) behave.
Uses the procedural demo rover, so nothing third-party is needed.
"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np
import trimesh

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SCRIPT = os.path.join(ROOT, 'scripts', 'blueprint3d.py')
DEMO = os.path.join(ROOT, 'assets', 'demo', 'demo_rover.glb')
SPEC = os.path.join(ROOT, 'assets', 'example_spec_3d.json')
TMP = tempfile.mkdtemp(prefix='bp_test_')
ENV = dict(os.environ, BP_CACHE=os.path.join(TMP, 'cache'))
passed = []


def run(model, spec, *extra):
    p = subprocess.run([sys.executable, SCRIPT, 'build', model, spec, '--out', os.path.join(TMP, 'out.png'), *extra],
                       capture_output=True, text=True, env=ENV)
    return p.returncode, p.stdout + p.stderr


def fresh_spec(name, **over):
    sp = json.load(open(SPEC)); sp.update(width=960, height=540, **over)
    path = os.path.join(TMP, name + '.json'); json.dump(sp, open(path, 'w'))
    lock = os.path.join(TMP, name + '.lock.json')
    if os.path.exists(lock):
        os.remove(lock)
    return path, lock


def check(name, cond, detail=''):
    if not cond:
        print(f'FAIL {name}\n{detail}'); sys.exit(1)
    passed.append(name); print(f'PASS {name}')


def export_variant(name, fn, fmt='glb'):
    m = trimesh.load(DEMO, process=False)
    m = m.to_geometry() if hasattr(m, 'to_geometry') else m.dump(concatenate=True)
    fn(m)
    path = os.path.join(TMP, f'{name}.{fmt}'); m.export(path)
    return path


spec, lock = fresh_spec('s')
rc, out = run(DEMO, spec)
check('first build locks the anchors', rc == 0 and os.path.exists(lock) and 'locked 7 anchors' in out, out)
lock_text = open(lock).read()
rc, out = run(DEMO, spec)
check('rebuild with the same model passes, lock untouched', rc == 0 and open(lock).read() == lock_text and 'binding:' not in out, out)

# lossless re-export of the same geometry (STL): must stay attached (pass) - or fail explicitly, never silently drift
stl = export_variant('same_geometry', lambda m: None, 'stl')
rc, out = run(stl, spec, '--up=+Y', '--front=-X')
check('lossless re-export stays attached (or fails explicitly)', rc == 0 or 'not bound' in out, out)

# different model: stretched along its length -> anchors move
def stretch(m):
    m.apply_transform(np.diag([1.5, 1, 1, 1]))


stretched = export_variant('stretched', stretch)
rc, out = run(stretched, spec)
check('replaced (stretched) model is rejected explicitly', rc != 0 and 'not bound' in out and 'geometry is different' in out, out)

# different export orientation of the same model: re-oriented file with unchanged spec settings
def rotate(m):
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))


rot = export_variant('rotated', rotate)
rc, out = run(rot, spec)
check('re-oriented model is rejected explicitly', rc != 0 and 'not bound' in out, out)

# same file, changed orientation setting
rc, out = run(DEMO, spec, '--front=+X')
check('changed --front is rejected and named in the message', rc != 0 and 'front -X -> +X' in out, out)

# changed view in the spec (cand anchors depend on it)
spec_v, lock_v = fresh_spec('v')
run(DEMO, spec_v)
sp = json.load(open(spec_v)); sp['view'] = {'az': 30, 'el': 35}; json.dump(sp, open(spec_v, 'w'))
rc, out = run(DEMO, spec_v)
check('changed view az/el is rejected and named in the message', rc != 0 and 'az 58 -> 30' in out, out)

# --relock accepts the new state, then a plain build passes
rc, out = run(DEMO, spec_v, '--relock')
check('--relock accepts the new settings', rc == 0 and 're-locked' in out, out)
rc, out = run(DEMO, spec_v)
check('build passes after --relock', rc == 0, out)

# intentional anchor edit on the same model/settings: lock entry updated silently
spec_e, lock_e = fresh_spec('e')
run(DEMO, spec_e)
sp = json.load(open(spec_e)); sp['callouts'][0]['cand'] = 6; json.dump(sp, open(spec_e, 'w'))
rc, out = run(DEMO, spec_e)
check('intentional anchor edit is accepted and re-locked', rc == 0 and 'locked 7 anchors' in out, out)
check('lock now records the edited anchor', json.load(open(lock_e))['callouts'][0]['anchor_def'] == {'cand': 6})

# escape hatch
rc, out = run(stretched, spec, '--no-lock')
check('--no-lock skips the check', rc == 0, out)

shutil.rmtree(TMP, ignore_errors=True)
print(f'\nall {len(passed)} checks passed')

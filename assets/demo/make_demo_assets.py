#!/usr/bin/env python3
"""Generate the demo assets used by the examples: a procedural rover model (GLB) and a flat-shaded demo image (PNG).

Everything is built from primitives in this script, so there is nothing third-party in the repo and anyone can
regenerate the files:   python3 assets/demo/make_demo_assets.py
Units are metres (y up, the rover's front faces -x, as expected by blueprint3d.py for GLB files).
"""
import os, sys
import numpy as np
import trimesh as tm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'scripts'))

T = tm.transformations
parts = []


def add(mesh, pos=(0, 0, 0), rot=None):
    m = mesh.copy()
    if rot is not None:
        m.apply_transform(rot)
    m.apply_translation(pos)
    parts.append(m)


def box(x, y, z):
    return tm.creation.box(extents=(x, y, z))


def cyl(r, h, sections=32):
    return tm.creation.cylinder(radius=r, height=h, sections=sections)   # axis = z


rot_x90 = T.rotation_matrix(np.pi / 2, [1, 0, 0])      # cylinder axis z -> y
rot_y90 = T.rotation_matrix(np.pi / 2, [0, 1, 0])      # cylinder axis z -> x

# chassis and deck
add(box(1.00, 0.26, 0.52), (0.05, 0.43, 0))
add(box(1.10, 0.05, 0.46), (0.05, 0.27, 0))
add(box(0.60, 0.02, 0.40), (0.15, 0.575, 0), T.rotation_matrix(np.radians(-6), [0, 0, 1]))   # solar panel
# six wheels with hubs and grousers, rocker arms
for x in (-0.55, 0.0, 0.55):
    for z in (-0.38, 0.38):
        add(cyl(0.17, 0.14), (x, 0.17, z))
        add(cyl(0.06, 0.18), (x, 0.17, z))
        for k in range(12):
            a = 2 * np.pi * k / 12
            g = box(0.05, 0.02, 0.14)
            g.apply_transform(T.rotation_matrix(a, [0, 0, 1]))
            add(g, (x + 0.18 * np.cos(a), 0.17 + 0.18 * np.sin(a), z))
for z in (-0.33, 0.33):
    add(box(1.15, 0.035, 0.045), (0.0, 0.29, z))
    for x in (-0.55, 0.0, 0.55):
        add(box(0.04, 0.12, 0.04), (x, 0.23, z * 1.0))
# mast with camera head (front, -x)
add(cyl(0.025, 0.5), (-0.35, 0.80, 0.15), rot_x90)
add(box(0.16, 0.10, 0.22), (-0.35, 1.08, 0.15))
for dz in (-0.06, 0.06):
    add(cyl(0.032, 0.05), (-0.45, 1.08, 0.15 + dz), rot_y90)
# high-gain dish (rear)
add(tm.creation.cone(radius=0.20, height=0.09, sections=32), (0.45, 0.72, -0.12), T.rotation_matrix(np.radians(-55), [0, 1, 0]))
add(cyl(0.012, 0.16), (0.45, 0.66, -0.12), rot_x90)
# whip antenna
add(cyl(0.008, 0.55), (0.50, 0.86, 0.20), rot_x90)
# front manipulator arm
add(box(0.32, 0.04, 0.05), (-0.62, 0.40, -0.14), T.rotation_matrix(np.radians(25), [0, 0, 1]))
add(box(0.22, 0.035, 0.045), (-0.80, 0.30, -0.14), T.rotation_matrix(np.radians(-30), [0, 0, 1]))
add(box(0.06, 0.07, 0.06), (-0.90, 0.25, -0.14))

scene = tm.Scene(parts)
glb = os.path.join(HERE, 'demo_rover.glb')
scene.export(glb)
m = scene.to_geometry() if hasattr(scene, 'to_geometry') else scene.dump(concatenate=True)
print(f'wrote {glb}: {len(m.faces)} faces, extents (x,y,z) = {np.round(m.extents, 3).tolist()} m')

# ---- flat demo image (shaded subject on a plain light background) for the 2D pipeline
from PIL import Image
from hlr import load_mesh, Prep, render
mesh = load_mesh(glb, '+Y', '-X')
prep = Prep(mesh, 38)
el, az = np.radians(18), np.radians(52)
cam = np.array([-np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
pal = dict(fill_lo=(70, 72, 78), fill_hi=(225, 225, 230), feat=(40, 40, 45), sil=(20, 20, 22))
img, _, _ = render(prep, -cam, [0, 1, 0], 1600, 900, 0.07, pal=pal, lw_feat=1, lw_sil=2)
bg = Image.new('RGBA', img.size, (238, 238, 236, 255)); bg.alpha_composite(img)
png = os.path.join(HERE, 'demo_image.png')
bg.convert('RGB').save(png)
print('wrote', png)

# blueprint-maker

![Python](https://img.shields.io/badge/python-3-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Turns a 3D model (GLB, glTF, OBJ, STL, PLY, OFF) or a 2D image into a technical blueprint sheet (PNG, 1920x1080 by default): hidden-line views, dimension lines, numbered callouts, description, spec table and title block. Works as a [Claude skill](#as-a-claude-skill) or as a standalone command-line tool.

- **3D model:** large 3/4 view plus side, top and front views with measured dimensions.
- **2D image:** line extraction with silhouette outline, bounding-box dimensions and callouts (single view only).
- **Callouts:** label positions are chosen automatically; a layout audit reports overlaps and crossing leaders.
- **Output:** three palettes (`blueprint`, `graphite`, `paper`), any 16:9 resolution, configurable top safe zone for desktop bars.

## Requirements

Python 3 (developed on 3.12).

```bash
pip install trimesh numpy scipy networkx pillow opencv-python-headless matplotlib
# optional, only for meshes above ~700k faces:
pip install fast-simplification
```

## Usage

```bash
# 3D model: inspect once (orientation + numbered anchor candidates), then build
python3 scripts/blueprint3d.py inspect model.glb --out insp
python3 scripts/blueprint3d.py build model.glb spec.json --out blueprint.png

# 2D image
python3 scripts/blueprint2d.py inspect image.png --out insp
python3 scripts/blueprint2d.py build image.png spec.json --out blueprint.png
```

`inspect` writes a single `inspect.png` to find the model's up/front axes (`--up`, `--front`) and the anchor candidates used by `"cand"` in the spec.

Minimal 3D `spec.json`:

```json
{
  "title": "My Object",
  "subtitle": "Short subtitle",
  "description": "Two or three sentences describing what is visible in the render.",
  "specs": [["Material", "Alloy"], ["Drive", "Electric"]],
  "notes": ["First remark.", "Second remark."],
  "callouts": [
    {"title": "Front sensor", "desc": "One short sentence.", "cand": 1},
    {"title": "Rear module", "desc": "One short sentence.", "cand": 13}
  ]
}
```

Useful keys: `width`/`height`, `top_safe`, `palette`, `up`/`front`, `view` (`az`/`el`), `layout` (`multi`/`single`), `scale_ref` (real-world size for dimensions). Without `scale_ref`, dimensions are in model units (`u`) or pixels (`px`). Full reference in [`SKILL.md`](SKILL.md); working examples in [`assets/`](assets/).

### Generating `spec.json` locally using an LLM
When using the tool standalone in local mode, you can quickly generate a complete `spec.json` file by prompting any LLM (ChatGPT, Claude, Gemini, etc.). See [`PROMPT_SPEC_TEMPLATE.md`](PROMPT_SPEC_TEMPLATE.md) for the ready-to-use prompt template.

Painted views are cached in `~/.cache/blueprint-maker` (or `$BP_CACHE`), so editing text or callouts rebuilds in seconds. Use `--no-cache` to disable.

## As a Claude skill

Install the packaged `blueprint-maker.skill` file (or copy this folder into your skills directory). Claude then runs the workflow above whenever you ask for a blueprint, technical drawing or technical wallpaper of a model or image.

## Limitations

- A 2D image has one viewpoint: no real front/side/top views. Use a 3D model for those.
- Line density depends on the mesh or image; very dense game meshes give busy lines.
- Dimensions are not real-world unless `scale_ref` is given.
- Text rendering covers Latin, Greek and Cyrillic only. The layout is designed for 16:9.
- The 2D background mask works for flat or transparent backgrounds; otherwise supply `"bg": "none"`, a crop, or a mask image.

## Project structure

```
blueprint-maker/
├── SKILL.md              # skill instructions + full spec reference
├── scripts/
│   ├── blueprint3d.py    # 3D CLI (inspect, build)
│   ├── blueprint2d.py    # 2D CLI (inspect, build)
│   ├── hlr.py            # hidden-line renderer, render cache
│   └── bp_common.py      # layout, drawing, label placement and audit
└── assets/               # example specs
```

## License

MIT. Models and images you convert remain subject to their own licenses; do not redistribute assets you do not own.

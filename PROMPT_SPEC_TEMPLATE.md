# Generating `spec.json` with an AI Assistant

When running `blueprint-maker` locally from the command line without Claude orchestrating the skill, you need a `spec.json` file to run the `build` command.

You can use any LLM (Claude, ChatGPT, Gemini, etc.) to generate this JSON for you using the prompt template below. This template covers **3D models**; for 2D images the callouts use `point_px` instead of `cand` (see `SKILL.md`).

---

## Workflow

1. **Inspect your model:**
   ```bash
   python3 scripts/blueprint3d.py inspect path/to/model.glb --out insp
   ```
   For formats other than GLB/glTF you usually need `--up +Z`; use `=` for negative axes (e.g. `--front=-X`). The terminal prints the model extents and one line per anchor candidate, e.g. `cand 3: pick=[581, 150] canonical=[-0.475, 1.11, 0.222] rel=[0.27, 0.98, 0.74]`. It also writes `insp/inspect.png` with the candidates numbered on the model.

2. **Check the orientation** in `insp/inspect.png`: the object must be upright and its front must point to the left of the 3/4 view. If not, re-run `inspect` with other `--up` / `--front` values.

3. **Copy the prompt below** and fill in the placeholders:
   - Object name and, if you have one, a description (the more you give, the less the AI has to guess).
   - The `up` / `front` values you used, and `az` / `el` if you changed them.
   - Paste the `cand N: ...` lines. If your AI can see images, also attach `insp/inspect.png`.

4. **Save the generated JSON** as `spec.json`.

5. **Build the blueprint:**
   ```bash
   python3 scripts/blueprint3d.py build path/to/model.glb spec.json --out blueprint.png
   ```
   Read any `WARNING:` lines printed by the build (text too long, overlapping labels). The first build also writes `spec.lock.json`, which ties the callout anchors to this model: if you later change the model, orientation or view, the build stops with an explicit error instead of silently pointing at the wrong parts.

6. **Verify the callouts yourself.** A text-only AI cannot see the model, so it chooses `cand` numbers from their `rel` positions. Open `blueprint.png` and check that every dot lands on the part its label names; edit the `cand` number or delete the callout if not.

---

## Prompt Template

Copy and paste everything below into your AI chat:

```text
Generate a valid JSON file named "spec.json" for the blueprint-maker CLI tool (3D mode).

INPUT
- Object name: [INSERT OBJECT NAME]
- Description / details (use ONLY this information, do not add facts): [INSERT DESCRIPTION, OR WRITE "none"]
- Language: [English / Italian / etc.] (Latin, Greek or Cyrillic alphabets only)
- Orientation used for inspect: up = [+Y], front = [-X]
- Inspect view: az = [58], el = [20]   (keep these defaults unless you passed --az/--el)
- Anchor candidates: paste the "cand N: ..." lines printed by inspect, or write "none"

OUTPUT FORMAT
Return ONLY one ```json code block, with no other text. Valid JSON: no comments, no trailing commas.

SCHEMA
{
  "title": "Main object title",
  "subtitle": "Short subtitle or model designation",
  "designation": "Full designation or class",
  "class": "General category",
  "rev": "01",
  "width": 1920,
  "height": 1080,
  "top_safe": 76,
  "palette": "blueprint",
  "up": "<copy the up value from INPUT>",
  "front": "<copy the front value from INPUT>",
  "view": { "az": <from INPUT>, "el": <from INPUT> },
  "description": "2-3 sentences: purpose and key visible features.",
  "specs": [
    ["Label", "Value"]
  ],
  "notes": [
    "Short design remark or operational constraint."
  ],
  "callouts": [
    { "title": "Part name", "desc": "One short sentence.", "cand": 1 }
  ]
}

TEXT LIMITS (longer text gets shrunk or rejected by the layout)
- description: at most 650 characters.
- specs: 5 to 7 rows; label at most 12 characters, value at most 45 characters.
- notes: 3 to 4 items, each at most 90 characters.
- callout title: 2 to 4 words; callout desc: one sentence, at most 75 characters.

CONTENT RULES
- Use only what the object name and description support. Do not invent precise numbers (mass, speed, power, dimensions): dimensions are measured by the tool from the model. If the description is "none", keep all text generic and qualitative.
- Do not claim facts about a real-world product unless they are in the description.

CALLOUT RULES
- Position of the candidates: rel=[x, y, z] runs from 0 to 1 over the object. x: 0 = front end, 1 = rear end. y: 0 = lowest, 1 = highest. z: across the width (side is arbitrary).
- Choose "cand" numbers by comparing the rel values of the candidates (canonical=[...] is the same position in model units). Use only numbers listed in INPUT and each number at most once. Never invent a number.
- Create a callout only if the object name or description gives a reasonable basis for which part sits at that position (for example: a camera at the front and high up, wheels at the lowest points). If you are not sure, leave it out: 4 to 7 reliable callouts are better than many guesses.
- If the candidates are "none": omit the "callouts" key entirely. Do not copy the example cand number from the schema.
```

---

## Notes

- The AI is guessing which part is at which coordinates. Treat its callouts as a draft and verify them on the rendered image (step 6).
- All descriptive text (description, specs, notes) is generated content: review it, especially if you publish the result.

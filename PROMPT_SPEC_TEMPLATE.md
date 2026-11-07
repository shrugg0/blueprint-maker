# Generating `spec.json` with an AI Assistant

When running `blueprint-maker` locally from the command line without Claude automatically orchestrating the skill, you need a `spec.json` file to run the `build` command.

You can use any LLM (Claude, ChatGPT, Gemini, etc.) to generate this JSON for you using the prompt template below.

---

## Workflow

1. **Inspect your model:**
   ```bash
   python3 scripts/blueprint3d.py inspect path/to/model.glb --out insp
   ```
   The terminal will print orientation information (`up`, `front`) and candidate points (`cand 1` to `cand 16`).

2. **Copy the Prompt below** and replace the placeholders:
   - Object name and short description.
   - Paste the terminal output lines with the `cand` numbers (optional but recommended for accurate callout arrows).

3. **Save the generated JSON** into `spec.json` in your project folder.

4. **Build the blueprint:**
   ```bash
   python3 scripts/blueprint3d.py build path/to/model.glb spec.json --out blueprint.png
   ```

---

## Prompt Template

Copy and paste everything below into your AI chat:

```text
Generate a valid, raw JSON file named "spec.json" for the blueprint-maker CLI tool.

Input Details:
- Object Name: [INSERT OBJECT / VEHICLE NAME]
- Optional Description / Details: [INSERT BRIEF DESCRIPTION OR LEAVE BLANK FOR AUTO-GENERATION]
- Preferred Language: [English / Italian / etc.]

Inspect Output (paste below if you ran `python3 scripts/blueprint3d.py inspect <model>`):
[PASTE TERMINAL OUTPUT WITH CANDIDATE POINTS cand 1..16 HERE, OR REMOVE THIS LINE]

Strict Output Requirements:
1. Return ONLY the raw valid JSON inside a single ```json markdown code block, with no conversational filler.
2. Follow this schema strictly:

{
  "title": "Main Object Title",
  "subtitle": "Short subtitle or model designation",
  "designation": "Full technical designation or class",
  "class": "General category / classification",
  "rev": "01",
  "width": 1920,
  "height": 1080,
  "top_safe": 76,
  "palette": "blueprint",
  "up": "+Y",
  "front": "-X",
  "view": {
    "az": 58,
    "el": 20
  },
  "description": "2 to 3 technical sentences describing the object, its purpose, and key visible features.",
  "specs": [
    ["Specification Label 1", "Value 1"],
    ["Specification Label 2", "Value 2"],
    ["Specification Label 3", "Value 3"],
    ["Specification Label 4", "Value 4"],
    ["Specification Label 5", "Value 5"]
  ],
  "notes": [
    "Technical design remark or operational constraint.",
    "Material or manufacturing remark."
  ],
  "callouts": [
    {
      "title": "Subsystem / Part Name",
      "desc": "Short sentence explaining what this component does.",
      "cand": 1
    }
  ]
}

Callout Guidelines:
- If inspect candidate lines were supplied, map 4 to 8 callouts to the most appropriate candidate indices (`cand`).
- Keep callout titles concise (2-4 words) and descriptions to 1 clear sentence.
```

import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
meta=json.loads((root/"model/metadata.json").read_text())
rows=["| Feature | Type | Allowed range / values | Why it matters |","|---|---|---|---|"]
for name,spec in meta["features"].items():
    rows.append(f'| `{name}` | {spec["type"]} | {spec["min"]} – {spec["max"]} | {spec["why"]} |')
rows.append("| `target` | binary | `{0, 1}` | 0 = BENIGN; 1 = ATTACK; prediction label |")
table="\n".join(rows)
(root/"DATA_CONTRACT.md").write_text(table+"\n")
readme=(root/"README.md").read_text()
start="<!-- DATA_CONTRACT_START -->"
end="<!-- DATA_CONTRACT_END -->"
if start in readme and end in readme:
    before, rest=readme.split(start,1)
    _, after=rest.split(end,1)
    readme=before+start+"\n"+table+"\n"+end+after
    (root/"README.md").write_text(readme)
print("Updated README.md and DATA_CONTRACT.md with all features.")

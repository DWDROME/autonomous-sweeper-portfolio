"""Print historical metrics directly from the exported evidence JSON."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1] / "evidence"
def read(name):
    return json.loads((root / name).read_text(encoding="utf-8"))

for name in ("holdout", "online"):
    row = read(f"localization/{name}-ndt.json")
    print(f"NDT {name}: lateral P95={row['errors']['lateral_m']['p95']*100:.2f} cm; "
          f"accepted={row['accepted']}/{row['frames']}")
for method in "ABC":
    row = read(f"coverage/G3-{method}.json")
    print(f"4WS offline G3 {method}: coverage={row['coverage']['coverage_ratio']*100:.3f}%; "
          f"elapsed={row['motion']['elapsed_s']:.2f} s")
row = read("driving/scenario-result.json")
print(f"CARLA measured-curb driving: passed={row['passed']}; "
      f"truth progress={row['final_progress_m']:.4f} m; collisions={len(row['collisions'])}")
for row in read("cleaning/result.json")["cases"]:
    print(f"CARLA task {row['name']}: passed={row['passed']}; state={row['state']}; "
          f"full-region coverage={row['coverage']['coverage_ratio']*100:.2f}%")

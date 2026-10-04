"""Synthetic 12 cm curb with a 0.4 m observation gap; no CARLA dependency."""
from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perception.curb_geometry import detect_curb

x, y = np.meshgrid(np.arange(.0125, 1.9876, .025),
                   np.arange(.5125, 2.4876, .025), indexing="ij")
x, y = x.ravel(), y.ravel()
height = np.where(y >= 1.5, .12, 0.)
points = np.column_stack((x, y, 1.4 + .03*x + .02*y + height))
keep = ~((x > .8) & (x < 1.2))
result = detect_curb(points[keep], (height == 0)[keep], (0., 2., .5, 2.5), 1,
                     sensor_origin=(0., .75, .4))
print(json.dumps(dict(scope="synthetic geometry demonstration", reason=result.reason,
    segments=[dict(type=s.boundary_type, vertices=len(s.xyz),
                   x_range_m=[float(s.xyz[0, 0]), float(s.xyz[-1, 0])],
                   median_sigma_m=float(np.median(s.sigma_m))) for s in result.segments]), indent=2))

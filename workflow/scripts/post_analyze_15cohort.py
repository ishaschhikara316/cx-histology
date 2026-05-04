"""Run calibration / subgroup / outlier / mutation_correlation on the
15-cohort run by monkey-patching BEST_DIR. Each analysis runs in its
own subprocess so we get parallel CPU + isolated output streams."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/ISHA/cx-histology")
os.chdir(ROOT)

D15 = "results/cv/uni_15cohort_final"

# A small driver module that monkey-patches BEST_DIR before importing main()
DRIVER = """
import sys, os
sys.path.insert(0, '/root/ISHA/cx-histology')
os.chdir('/root/ISHA/cx-histology')
import workflow.scripts.{mod} as m
from pathlib import Path
m.BEST_DIR = Path('{best}')
m.main()
"""

ANALYSES = [
    ("calibration_analysis", "calibration"),
    ("subgroup_analysis",    "subgroup"),
    ("outlier_analysis",     "outlier"),
    ("mutation_correlation", "mutation_corr"),
]

procs = []
for mod, tag in ANALYSES:
    code = DRIVER.format(mod=mod, best=D15)
    log = open(f"logs/post15_{tag}.log", "w")
    p = subprocess.Popen([sys.executable, "-c", code],
                         stdout=log, stderr=subprocess.STDOUT,
                         env={**os.environ,
                              "PYTHONPATH": "/root/ISHA/cx-histology"})
    procs.append((tag, p, log))
    print(f"  launched {tag} (pid={p.pid})", flush=True)

# Wait
fails = []
for tag, p, log in procs:
    rc = p.wait()
    log.close()
    print(f"  {tag} rc={rc}", flush=True)
    if rc != 0:
        fails.append(tag)
print(f"\npost-analyze 15-cohort: {len(ANALYSES)-len(fails)}/{len(ANALYSES)} succeeded "
      f"(fails={fails})")

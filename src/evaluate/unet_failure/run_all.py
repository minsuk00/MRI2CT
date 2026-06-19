"""Run the full U-Net failure-anatomy pipeline: extract -> aggregate -> figures -> report."""
import os
import sys
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
# 05 per-label/region diagnosis, then 06 bone deep-dive
for mod in ["extract.py", "aggregate.py", "build_figures.py", "report.py",
            "bone_extract.py", "bone_aggregate.py", "mr_tissue.py", "bone_figures.py", "bone_report.py"]:
    print(f"\n===== {mod} =====", flush=True)
    sys.argv = [mod]
    runpy.run_path(os.path.join(HERE, mod), run_name="__main__")

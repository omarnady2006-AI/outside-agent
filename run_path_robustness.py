import pandas as pd
import os
import sys
import subprocess

def run_test(cwd, pythonpath):
    print(f"--- Running from: {cwd} ---")
    script = """
import pandas as pd
from leakage_agent import LeakageValidator
v = LeakageValidator()
res = v.validate(pd.DataFrame({"label":["cat"]}), copy_id="path_test")
print("decision:", res.decision)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    result = subprocess.run([sys.executable, "-c", script], cwd=cwd, env=env, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

# G1: From C:\Users\Admin\college
college_dir = r"C:\Users\Admin\college"
project_root = r"C:\Users\Admin\college\outside-agent"
pythonpath1 = os.path.join(project_root, "leakage_agent", "src")

run_test(college_dir, pythonpath1)

# G2: From project root again
run_test(project_root, pythonpath1)

import subprocess
import os
import sys

# Task C: Run verbose smoke test subset
print("Running Task C: Verbose smoke test subset...")
env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath("leakage_agent/src")

cmd = [sys.executable, "-m", "pytest", "-vv", "-x", "leakage_agent/tests/test_pipeline.py"]
result = subprocess.run(cmd, env=env, capture_output=True, text=True)

print(result.stdout)
print(result.stderr)

if result.returncode != 0:
    print(f"Tests failed with exit code {result.returncode}")
    sys.exit(result.returncode)
else:
    print("Tests passed.")

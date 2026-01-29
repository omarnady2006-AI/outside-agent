import subprocess
import os
import sys

# Task B: Run unit tests
print("Running Task B: Unit tests...")
env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath("leakage_agent/src")

# Ensure outputs directory exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

cmd = [sys.executable, "-m", "pytest", "-q", "leakage_agent/tests"]
result = subprocess.run(cmd, env=env, capture_output=True, text=True)

print(result.stdout)
print(result.stderr)

with open("outputs/test_report_unit.txt", "w") as f:
    f.write(result.stdout)
    f.write(result.stderr)

if result.returncode != 0:
    print(f"Tests failed with exit code {result.returncode}")
    sys.exit(result.returncode)
else:
    print("Tests passed.")

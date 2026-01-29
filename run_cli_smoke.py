import subprocess
import os
import sys

# Task E: Run CLI smoke test
print("Running Task E: CLI smoke test...")
env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath("leakage_agent/src")

# Create tmp_smoke directory
tmp_dir = "tmp_smoke"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# Create ok.csv
with open(os.path.join(tmp_dir, "ok.csv"), "w") as f:
    f.write("email,label,amount\nalice@example.com,cat,100")

# Run CLI for ok.csv
print("Testing ok.csv...")
cmd_ok = [sys.executable, "-m", "leakage_agent.cli", "run", "--input", os.path.join(tmp_dir, "ok.csv"), "--copy-id", "smoke_cli_ok"]
res_ok = subprocess.run(cmd_ok, env=env, capture_output=True, text=True)
print(res_ok.stdout)
print(res_ok.stderr)
print(f"EXIT={res_ok.returncode}")

# Create secret.csv
with open(os.path.join(tmp_dir, "secret.csv"), "w") as f:
    f.write("secret_key,label\nmy-secret-token,dog")

# Run CLI for secret.csv
print("\nTesting secret.csv...")
cmd_secret = [sys.executable, "-m", "leakage_agent.cli", "run", "--input", os.path.join(tmp_dir, "secret.csv"), "--copy-id", "smoke_cli_secret"]
res_secret = subprocess.run(cmd_secret, env=env, capture_output=True, text=True)
print(res_secret.stdout)
print(res_secret.stderr)
print(f"EXIT={res_secret.returncode}")

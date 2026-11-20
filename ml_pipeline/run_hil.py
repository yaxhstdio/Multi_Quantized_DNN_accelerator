import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["int16", "int8", "int4", "mixed"], required=True)
args = parser.parse_args()

# 1. Run export_all_modes.py and capture the printed vvp command
exp = subprocess.run(
    [sys.executable, "export_all_modes.py", "--mode", args.mode],
    capture_output=True, text=True, check=True
)
print(exp.stdout)

vvp_cmd = None
for line in exp.stdout.splitlines():
    if line.strip().startswith("vvp sim_hil.vvp"):
        vvp_cmd = line.strip()

if not vvp_cmd:
    raise RuntimeError("Could not find vvp command in export_all_modes.py output.")

# 2. Copy all fresh .hex files to WSL and run sim_hil.vvp
wsl_cmd = (
    'cp "/mnt/c/Users/meyas/Desktop/study material/projects/MLCV + DNN accelerator/"*.hex ~/EDI/ && '
    f'cd ~/EDI && {vvp_cmd}'
)
sim = subprocess.run(["wsl", "bash", "-c", wsl_cmd], capture_output=True, text=True, check=True)
print(sim.stdout)
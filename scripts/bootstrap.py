from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Meteor Quant reproducibly.")
    parser.add_argument("--dev", action="store_true", help="install test, lint, and build tools")
    parser.add_argument("--marketlm", action="store_true", help="install MarketLM dependencies")
    parser.add_argument("--markethybrid", action="store_true", help="install MarketHybrid dependencies")
    parser.add_argument("--timesfm", action="store_true", help="install TimesFM dependencies")
    parser.add_argument("--ml", action="store_true", help="install all optional ML dependencies")
    parser.add_argument("--frontend", action="store_true", help="rebuild the React dashboard")
    parser.add_argument("--rust", action="store_true", help="build the native Rust engine")
    parser.add_argument("--no-editable", action="store_true", help="install a regular package instead of editable mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sys.version_info < (3, 11):  # noqa: UP036
        raise SystemExit("Meteor Quant requires Python 3.11 or newer.")
    if sys.maxsize <= 2**32:
        raise SystemExit("Meteor Quant requires 64-bit Python.")

    if not venv_python().exists():
        run([sys.executable, "-m", "venv", str(VENV)])

    python = str(venv_python())
    run([python, "-m", "pip", "install", "--upgrade", "pip"])

    extras: list[str] = []
    if args.dev:
        extras.append("dev")
    if args.ml:
        extras.append("ml")
    else:
        if args.marketlm:
            extras.append("marketlm")
        if args.markethybrid:
            extras.append("markethybrid")
        if args.timesfm:
            extras.append("timesfm")

    spec = "." + (f"[{','.join(extras)}]" if extras else "")
    command = [python, "-m", "pip", "install"]
    if not args.no_editable:
        command.append("-e")
    command.append(spec)
    run(command)

    if args.frontend:
        npm = shutil.which("npm")
        if npm is None:
            raise SystemExit("npm is required for --frontend. Install Node.js 22+ or use the prebuilt bundle.")
        run([npm, "ci", "--no-audit", "--no-fund"], cwd=ROOT / "frontend")
        run([npm, "run", "build"], cwd=ROOT / "frontend")
        run([python, str(ROOT / "scripts" / "sync_frontend.py")])

    if args.rust:
        cargo = shutil.which("cargo")
        if cargo is None:
            raise SystemExit("cargo is required for --rust. Install Rust with rustup.")
        run([cargo, "build", "--release"], cwd=ROOT / "rust" / "meteor-engine")

    print("\nMeteor Quant is installed.")
    print("Run ./run.sh on Linux/macOS or .\\run.ps1 on Windows.")


if __name__ == "__main__":
    main()

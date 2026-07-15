from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "frontend" / "dist"
DESTINATION = ROOT / "src" / "meteor_quant" / "static"


def main() -> None:
    index = SOURCE / "index.html"
    if not index.is_file():
        raise SystemExit("frontend/dist is missing; run the frontend build first")
    temporary = DESTINATION.with_name(f"{DESTINATION.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(SOURCE, temporary)
    shutil.rmtree(DESTINATION, ignore_errors=True)
    temporary.replace(DESTINATION)
    print(f"Synced frontend bundle to {DESTINATION}")


if __name__ == "__main__":
    main()

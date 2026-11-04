from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from myownai.hardware import get_hardware_report

def main() -> None:
    report = get_hardware_report()
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "HARDWARE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

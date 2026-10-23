"""Write (or --check) data/MANIFEST.json: SHA-256 of every frozen raw file."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MAN = ROOT / "data" / "MANIFEST.json"


def digest() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.iterdir()) if p.is_file()}


if __name__ == "__main__":
    if "--check" in sys.argv:
        want = json.loads(MAN.read_text())
        got = digest()
        bad = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
        if bad:
            sys.exit(f"MANIFEST mismatch: {bad}")
        print(f"manifest OK ({len(got)} files)")
    else:
        MAN.write_text(json.dumps(digest(), indent=2) + "\n")
        print(f"wrote {MAN}")

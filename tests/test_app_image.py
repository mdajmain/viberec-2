"""The app image copies only the files whitelisted in .dockerignore; every module they import must be among them."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_image_has_every_imported_module():
    allowed = {l[1:].strip() for l in (ROOT / ".dockerignore").read_text().splitlines() if l.startswith("!")}
    in_image = {p for p in allowed if p.endswith(".py")}
    missing = []
    for f in [ROOT / "app" / "main.py"] + [ROOT / p for p in in_image]:
        text = f.read_text()
        mods = re.findall(r"^\s*from \.(\w+) import", text, re.M) + re.findall(r"from src\.viberec\.(\w+) import", text)
        for group in re.findall(r"^\s*from (?:\.|src\.viberec) import ([\w ,]+)", text, re.M):  # "from . import a, b as c"
            mods += [part.split(" as ")[0].strip() for part in group.split(",")]
        for mod in mods:
            if f"src/viberec/{mod}.py" not in in_image:
                missing.append(f"{f.name} -> {mod}")
    assert missing == []

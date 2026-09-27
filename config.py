"""Load and save config.yaml."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
_TOP_KEY = re.compile(r"^([A-Za-z_][\w-]*):")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("regions", {})
    cfg.setdefault("colors", {})
    cfg.setdefault("no_click_zones", {})
    return cfg


def update_sections(updates, path=CONFIG_PATH):
    """Replace whole top-level sections of config.yaml, leaving every other line
    (including comments) untouched. Missing sections are appended."""
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    for key, value in updates.items():
        block = yaml.safe_dump({key: value}, sort_keys=False, default_flow_style=None)
        start = next((i for i, line in enumerate(lines)
                      if (m := _TOP_KEY.match(line)) and m.group(1) == key), None)
        if start is None:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines += ["\n"] + block.splitlines(keepends=True)
            continue
        end = start + 1
        while end < len(lines) and not _TOP_KEY.match(lines[end]):
            end += 1
        # Keep blank lines and comments that introduce the next section.
        while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].startswith("#")):
            end -= 1
        lines[start:end] = block.splitlines(keepends=True)
    text = "".join(lines)
    yaml.safe_load(text)  # never write a file that no longer parses
    Path(path).write_text(text, encoding="utf-8")

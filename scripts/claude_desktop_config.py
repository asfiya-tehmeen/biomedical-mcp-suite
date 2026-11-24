r"""Print a Claude Desktop config block for all five biomed-mcp servers (absolute paths, own cache per server).

  python scripts/claude_desktop_config.py            # print
  python scripts/claude_desktop_config.py --write    # merge into %APPDATA%\Claude\claude_desktop_config.json (backs up first)
Keys are read from the project's .env by the servers themselves; nothing secret is written to the config.
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVERS = ["trials", "labels", "openfda", "pubmed", "rxnorm"]


def block() -> dict:
    return {f"biomed-{n}": {"command": sys.executable, "args": ["-m", f"biomed_mcp.servers.{n}"],
                            "env": {"BIOMED_CACHE_DIR": str(ROOT / "data" / "cache")}} for n in SERVERS}


if __name__ == "__main__":
    cfg = {"mcpServers": block()}
    if "--write" not in sys.argv:
        print(json.dumps(cfg, indent=2))
        sys.exit()
    path = Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if path.exists():
        shutil.copy(path, path.with_suffix(".json.bak"))
    existing.setdefault("mcpServers", {}).update(cfg["mcpServers"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"updated {path} (backup: {path.with_suffix('.json.bak')}). Restart Claude Desktop.")

"""biomed-mcp: provenance-first MCP servers over public biomedical APIs."""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from <project>/.env or ./.env (real env vars win; empty values ignored)."""
    for path in (Path(__file__).resolve().parents[2] / ".env", Path(".env")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k.strip(), v)


_load_dotenv()

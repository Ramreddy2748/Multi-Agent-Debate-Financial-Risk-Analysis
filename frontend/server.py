"""
Compatibility launcher for the FastAPI stock-risk app.

Preferred production-style command:
    uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8766
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Missing API dependencies. Install them with:\n"
            "  pip install -r requirements.txt\n"
            "or:\n"
            "  pip install fastapi uvicorn\n"
            f"Original error: {exc}"
        )

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()

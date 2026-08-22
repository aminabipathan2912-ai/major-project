import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import uvicorn  # noqa: E402

from cctv_ai.demo.server import create_app  # noqa: E402


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()


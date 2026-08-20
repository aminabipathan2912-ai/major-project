"""
scripts/init_db.py
One-time database initialisation script.
Run from repo root: python scripts/init_db.py
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.database import init_db


async def main():
    print("Initialising database …")
    await init_db()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())

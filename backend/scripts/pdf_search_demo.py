#!/usr/bin/env python3
import os
import sys
import asyncio

CURRENT_DIR = os.path.dirname(__file__)
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


async def main_async():
    from test_pdf_search import test_pdf_vector_search  # type: ignore
    await test_pdf_vector_search()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


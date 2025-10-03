#!/usr/bin/env python3
"""
Demo: Enhanced PDF ingestion pipeline.
Usage:
  python backend/scripts/pdf_ingestion_demo.py --file /path/to/file.pdf
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(__file__)
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.services.document_processor import DocumentProcessor  # type: ignore


async def run_pdf_ingestion(file_path: str):
    print("🔍 Testing Enhanced PDF Document Ingestion Pipeline")
    print("=" * 70)

    if not os.path.exists(file_path):
        print(f"❌ PDF file not found at: {file_path}")
        return False

    dp = DocumentProcessor()
    try:
        result = await dp.process_file(
            file_path=file_path,
            agent_id=999,
            filename=os.path.basename(file_path),
            file_type="application/pdf",
            document_id=999,
            extra_metadata={"test_run": True},
        )

        status = result.get("status")
        chunk_count = result.get("chunk_count", 0)
        print(f"✅ Status: {status} | Chunks: {chunk_count}")
        if preview := result.get("preview"):
            print("📖 Preview:")
            print(preview[:300] + ("..." if len(preview) > 300 else ""))
        return status == "completed" and chunk_count > 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to a PDF file")
    args = parser.parse_args()
    ok = asyncio.run(run_pdf_ingestion(args.file))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


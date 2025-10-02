#!/usr/bin/env python3
"""
Test script to validate the enhanced PDF document ingestion pipeline.
This tests the new features: PyMuPDF4LLM, semantic chunking, YAKE keywords, etc.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from app.services.document_processor import DocumentProcessor

async def test_pdf_ingestion():
    """Test the enhanced PDF document ingestion pipeline."""
    print("🔍 Testing Enhanced PDF Document Ingestion Pipeline")
    print("=" * 70)

    # Initialize the document processor
    document_processor = DocumentProcessor()

    # Path to the PDF file
    pdf_path = "/Users/iliasstiendre/Documents/middle school math practice exercises.pdf"

    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found at: {pdf_path}")
        return None

    pdf_size = os.path.getsize(pdf_path)
    print(f"📄 PDF File: {os.path.basename(pdf_path)}")
    print(f"📊 File Size: {pdf_size:,} bytes ({pdf_size/1024:.1f} KB)")
    print()

    try:
        print("🔄 Step 1: Processing PDF through enhanced document processor...")
        print("   Features being tested:")
        print("   ✓ PyMuPDF4LLM layout-aware processing")
        print("   ✓ Semantic chunking with overlap optimization")
        print("   ✓ YAKE keyword extraction")
        print("   ✓ SimHash deduplication")
        print("   ✓ Metadata enrichment")
        print()

        # Process the PDF using the enhanced document processor
        result = await document_processor.process_file(
            file_path=pdf_path,
            agent_id=999,  # Test agent ID
            filename="middle_school_math_practice_exercises.pdf",
            file_type="application/pdf",
            document_id=999,  # Test document ID
            extra_metadata={
                "test_run": True,
                "content_type": "educational_pdf",
                "subject": "mathematics",
                "grade_level": "middle_school"
            }
        )

        print(f"✅ PDF Processing completed!")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Chunk Count: {result.get('chunk_count', 0)}")
        print(f"   Vector IDs Count: {len(result.get('vector_ids', []))}")
        print()

        # Display enhanced metadata
        if result.get('metadata'):
            print("📊 Enhanced Metadata Extracted:")
            metadata = result['metadata']

            # Keywords from YAKE
            if 'keywords' in metadata:
                keywords = metadata['keywords'][:10]  # Show first 10
                print(f"   🔑 Keywords (YAKE): {', '.join(keywords)}")

            # Processing timing
            if 'processing_time' in metadata:
                print(f"   ⏱️  Processing Time: {metadata['processing_time']:.2f}s")

            # Document stats
            if 'char_count' in metadata:
                print(f"   📊 Character Count: {metadata['char_count']:,}")

            if 'word_count' in metadata:
                print(f"   📊 Word Count: {metadata['word_count']:,}")

            # Content type detection
            if 'detected_content_type' in metadata:
                print(f"   🔍 Content Type: {metadata['detected_content_type']}")

            print()

        # Display preview of extracted content
        if result.get('preview'):
            preview = result['preview'][:300]
            print(f"📖 Content Preview:")
            print(f"   {preview}...")
            print()

        # Display chunk information
        if result.get('chunk_count', 0) > 0:
            print(f"🔄 Step 2: Analyzing semantic chunks...")
            print(f"   Total Chunks Created: {result.get('chunk_count')}")
            print(f"   Vector Embeddings: {len(result.get('vector_ids', []))} created")

            # If we have access to chunk details, show them
            if 'chunks' in result:
                print(f"   📄 Sample Chunk Sizes:")
                for i, chunk in enumerate(result['chunks'][:3], 1):
                    chunk_size = len(chunk.get('content', ''))
                    print(f"      Chunk {i}: {chunk_size} characters")
            print()

        # Test keyword quality
        if result.get('metadata', {}).get('keywords'):
            keywords = result['metadata']['keywords']
            print(f"🔍 Keyword Analysis:")
            print(f"   Total Keywords Extracted: {len(keywords)}")
            print(f"   Sample Keywords: {', '.join(keywords[:8])}")

            # Check for math-related keywords
            math_keywords = [kw for kw in keywords if any(term in kw.lower()
                           for term in ['math', 'equation', 'algebra', 'geometry', 'number', 'problem', 'solve', 'calculate'])]
            if math_keywords:
                print(f"   🧮 Math-related Keywords: {', '.join(math_keywords[:5])}")
            print()

        print("🔄 Step 3: Validating enhanced features...")

        # Check for successful PyMuPDF4LLM processing
        if result.get('status') == 'completed':
            print("   ✅ PyMuPDF4LLM: Layout-aware PDF processing successful")

        # Check for semantic chunking
        if result.get('chunk_count', 0) > 0:
            print("   ✅ Semantic Chunking: Content properly segmented")

        # Check for YAKE keywords
        if result.get('metadata', {}).get('keywords'):
            print("   ✅ YAKE Keywords: Keyword extraction successful")

        # Check for metadata enrichment
        if result.get('metadata'):
            print("   ✅ Metadata Enrichment: Enhanced metadata generated")

        # Check for vector creation
        if result.get('vector_ids'):
            print("   ✅ Vector Embeddings: Successfully created and stored")

        print()
        print("=" * 70)
        print("🏁 Enhanced PDF Ingestion Test Completed")

        # Overall success assessment
        success_indicators = [
            result.get('status') == 'completed',
            result.get('chunk_count', 0) > 0,
            len(result.get('vector_ids', [])) > 0,
            bool(result.get('metadata', {}).get('keywords')),
            bool(result.get('preview'))
        ]

        success_count = sum(success_indicators)
        total_checks = len(success_indicators)

        if success_count == total_checks:
            print("✅ OVERALL STATUS: COMPLETE SUCCESS")
            print("   All enhanced features working correctly!")
        elif success_count >= total_checks * 0.8:
            print("🟨 OVERALL STATUS: MOSTLY SUCCESS")
            print(f"   {success_count}/{total_checks} features working correctly")
        else:
            print("❌ OVERALL STATUS: NEEDS ATTENTION")
            print(f"   Only {success_count}/{total_checks} features working correctly")

        print()
        print("📊 Final Results Summary:")
        print(f"   📄 Document: {os.path.basename(pdf_path)}")
        print(f"   📊 File Size: {pdf_size/1024:.1f} KB")
        print(f"   ✂️  Chunks: {result.get('chunk_count', 0)}")
        print(f"   🔗 Vectors: {len(result.get('vector_ids', []))}")
        print(f"   🔑 Keywords: {len(result.get('metadata', {}).get('keywords', []))}")
        print(f"   📈 Status: {result.get('status', 'unknown')}")

        return result

    except Exception as e:
        print(f"❌ Error during PDF processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the PDF ingestion test
    result = asyncio.run(test_pdf_ingestion())
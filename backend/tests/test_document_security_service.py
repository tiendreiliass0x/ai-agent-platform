import pytest


@pytest.mark.asyncio
async def test_document_security_safe_text():
    from app.services.document_security import DocumentSecurityService

    svc = DocumentSecurityService()
    content = b"This is a safe text document with normal content."
    is_safe, details = await svc.validate_upload(
        file_content=content,
        filename="safe_document.txt",
        content_type="text/plain",
        user_id=1001,
    )

    assert is_safe is True
    assert details["file_info"]["extension"] == ".txt"
    assert details["file_info"]["mime_type"] == "text/plain"


@pytest.mark.asyncio
async def test_document_security_executable_blocked():
    from app.services.document_security import DocumentSecurityService

    svc = DocumentSecurityService()
    # PE header ("MZ") triggers signature/extension checks
    pe_bytes = b"\x4d\x5a" + b"EXE_CONTENT"
    is_safe, details = await svc.validate_upload(
        file_content=pe_bytes,
        filename="malware.exe",
        content_type="application/octet-stream",
        user_id=1002,
    )

    assert is_safe is False
    assert any("Disallowed file extension" in issue for issue in details.get("issues", [])) or any(
        "Executable file detected" in issue for issue in details.get("issues", [])
    )


@pytest.mark.asyncio
async def test_document_security_pdf_with_js_blocked():
    from app.services.document_security import DocumentSecurityService

    svc = DocumentSecurityService()
    # Minimal PDF header with a JavaScript object marker
    pdf_bytes = b"%PDF-1.4\n.../JS(app.alert('x'))"
    is_safe, details = await svc.validate_upload(
        file_content=pdf_bytes,
        filename="with_js.pdf",
        content_type="application/pdf",
        user_id=1003,
    )

    assert is_safe is False
    assert any("PDF contains JavaScript" in issue for issue in details.get("issues", []))


@pytest.mark.asyncio
async def test_document_security_empty_file_blocked():
    from app.services.document_security import DocumentSecurityService

    svc = DocumentSecurityService()
    is_safe, details = await svc.validate_upload(
        file_content=b"",
        filename="empty.txt",
        content_type="text/plain",
        user_id=1004,
    )

    assert is_safe is False
    assert any("Empty file" in issue for issue in details.get("issues", []))


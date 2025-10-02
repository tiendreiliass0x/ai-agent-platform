import pytest

from app.services.langextract_service import lang_extract_service, LangExtractResult


@pytest.mark.asyncio
async def test_langextract_returns_result_for_text():
    text = "Katana provides excellent order management for Shopify."
    result = await lang_extract_service.analyze_text(text)
    assert isinstance(result, LangExtractResult)
    assert result.sentiment["label"] in {"positive", "neutral", "negative"}
    assert isinstance(result.entities, list)


@pytest.mark.asyncio
async def test_langextract_none_for_empty_text():
    result = await lang_extract_service.analyze_text("   ")
    assert result is None

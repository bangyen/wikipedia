"""Integration tests for Wikipedia article maturity pipeline."""

import pytest
from wikipedia.wiki_client import WikiClient
from wikipedia.models.baseline import HeuristicBaselineModel
from wikipedia.features.extractors import all_features


class TestIntegrationPipeline:
    """Test the full pipeline from fetching to scoring."""

    @pytest.mark.integration
    def test_full_pipeline_smoke(self) -> None:
        """Smoke test for the full pipeline using a real article."""
        client = WikiClient()
        model = HeuristicBaselineModel()

        # Test with a well-known article
        title = "Python (programming language)"

        # 1. Fetch
        page_content = client.get_page_content(title)
        sections = client.get_sections(title)
        templates = client.get_templates(title)
        revisions = client.get_revisions(title, limit=5)
        backlinks = client.get_backlinks(title, limit=10)
        citations = client.get_citations(title, limit=10)

        article_data = {
            "title": title,
            "data": {
                "parse": page_content,
                "query": {
                    "pages": {page_content.get("pageid", "0"): page_content},
                    "sections": sections,
                    "templates": templates,
                    "revisions": revisions,
                    "backlinks": backlinks,
                    "extlinks": citations,
                },
            },
        }

        # 2. Extract
        features = all_features(article_data)
        assert len(features) > 0
        assert "content_length" in features

        # 3. Score
        result = model.calculate_maturity_score(article_data)
        assert "maturity_score" in result
        assert 0 <= result["maturity_score"] <= 100
        print(f"Integration test score for {title}: {result['maturity_score']}")

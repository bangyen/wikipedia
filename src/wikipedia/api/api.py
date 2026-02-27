#!/usr/bin/env python3
"""FastAPI server for Wikipedia article maturity scoring.

This module provides a REST API endpoint for scoring Wikipedia articles
using the heuristic baseline model. Returns JSON with score, band, and
top contributing features.
"""

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from wikipedia.models.baseline import HeuristicBaselineModel
from wikipedia.wiki_client import WikiClient


class ScoreResponse(BaseModel):
    """Response model for maturity score endpoint."""

    title: str
    score: float
    band: str
    top_features: Dict[str, float]
    pillar_scores: Dict[str, float]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="Wikipedia Maturity Scoring API",
    description="API for scoring Wikipedia article maturity using heuristic baseline model",
    version="1.0.0",
)

# Initialize model and client
model = HeuristicBaselineModel()
wiki_client = WikiClient()


def get_maturity_band(score: float) -> str:
    """Convert numeric score to maturity band.

    Args:
        score: Maturity score (0-100)

    Returns:
        Maturity band string
    """
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    elif score >= 20:
        return "Poor"
    else:
        return "Very Poor"


def fetch_article_data(title: str) -> Optional[Dict[str, Any]]:
    """Fetch Wikipedia article data for scoring.

    Args:
        title: Article title

    Returns:
        Article data dictionary or None if not found
    """
    try:
        # Fetch comprehensive article data using the simplified WikiClient interface
        page_content = wiki_client.get_page_content(title)
        sections = wiki_client.get_sections(title)
        templates = wiki_client.get_templates(title)
        revisions = wiki_client.get_revisions(title, limit=20)
        backlinks = wiki_client.get_backlinks(title, limit=50)
        citations = wiki_client.get_citations(title, limit=50)

        # Combine data into expected format for the model
        # Note: The model expects a specific nested structure which we maintain here
        # for compatibility, but we access the flattened results from WikiClient.
        article_data = {
            "title": title,
            "data": {
                "parse": page_content,
                "query": {
                    "pages": page_content,  # Historical quirk: content often placed here
                    "sections": sections,
                    "templates": templates,
                    "revisions": revisions,
                    "backlinks": backlinks,
                    "extlinks": citations,
                },
            },
        }

        return article_data

    except Exception as e:
        print(f"Error fetching data for {title}: {e}")
        return None

    except Exception as e:
        print(f"Error fetching data for {title}: {e}")
        return None


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Wikipedia Maturity Scoring API",
        "version": "1.0.0",
        "endpoints": {  # type: ignore
            "/score": "GET /score?title=<article_title>",
            "/docs": "API documentation",
            "/health": "Health check",
        },
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "model": "loaded"}


@app.get("/score", response_model=ScoreResponse)
async def score_article(
    title: str = Query(..., description="Wikipedia article title to score")
) -> ScoreResponse:
    """Score a Wikipedia article for maturity.

    Args:
        title: Wikipedia article title

    Returns:
        ScoreResponse with maturity score, band, and top features

    Raises:
        HTTPException: If article not found or scoring fails
    """
    try:
        # Fetch article data
        article_data = fetch_article_data(title)

        if not article_data:
            raise HTTPException(
                status_code=404,
                detail=f"Article '{title}' not found or could not be fetched",
            )

        # Calculate maturity score
        result = model.calculate_maturity_score(article_data)

        # Get top contributing features
        feature_importance = model.get_feature_importance(article_data)
        top_features = dict(list(feature_importance.items())[:5])  # Top 5 features

        # Determine maturity band
        maturity_score = result["maturity_score"]
        band = get_maturity_band(maturity_score)

        return ScoreResponse(
            title=title,
            score=maturity_score,
            band=band,
            top_features=top_features,
            pillar_scores=result["pillar_scores"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error scoring article '{title}': {str(e)}"
        )


@app.exception_handler(404)
async def not_found_handler(request: Any, exc: Any) -> JSONResponse:
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "detail": "The requested resource was not found",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Any, exc: Any) -> JSONResponse:
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    print("Starting Wikipedia Maturity Scoring API...")
    print("API Documentation: http://localhost:8002/docs")
    print("Health Check: http://localhost:8002/health")
    print("Score Endpoint: http://localhost:8002/score?title=<article_title>")

    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True, log_level="info")

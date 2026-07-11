"""Metadata filters for scoped Haystack retrieval."""

from __future__ import annotations

from models import RetrievalRequest


def build_chat_session_filter(request: RetrievalRequest) -> dict[str, object]:
    """Build a Haystack metadata filter scoped to the request chat and session."""
    return {
        "operator": "AND",
        "conditions": [
            {
                "field": "meta.chat_id",
                "operator": "==",
                "value": str(request.chat_id),
            },
            {
                "field": "meta.session_id",
                "operator": "==",
                "value": request.session_id,
            },
        ],
    }

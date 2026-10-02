"""Metadata filters for scoped Haystack retrieval."""

from __future__ import annotations

from models import RetrievalRequest


def build_session_filter(chat_id: int, session_id: str) -> dict[str, object]:
    """Build a Haystack metadata filter scoped to one chat and one session."""
    return {
        "operator": "AND",
        "conditions": [
            {
                "field": "meta.chat_id",
                "operator": "==",
                "value": str(chat_id),
            },
            {
                "field": "meta.session_id",
                "operator": "==",
                "value": session_id,
            },
        ],
    }


def build_chat_session_filter(request: RetrievalRequest) -> dict[str, object]:
    """Build a Haystack metadata filter scoped to the request chat and session."""
    return build_session_filter(request.chat_id, request.session_id)

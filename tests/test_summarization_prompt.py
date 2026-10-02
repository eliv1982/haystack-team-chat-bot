"""Tests for summarization prompt template rendering."""

from __future__ import annotations

from datetime import datetime, timezone

from haystack import Document
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses.chat_message import ChatRole

from documents import chat_message_to_document
from models import ChatMessage
from summarization_prompt import SUMMARIZATION_PROMPT_TEMPLATE


def _discussion_documents() -> list[Document]:
    first = chat_message_to_document(
        ChatMessage(
            chat_id=-1001234567890,
            message_id=42,
            user_id=7,
            session_id="chat:-1001234567890",
            author_name="Alice",
            username="alice",
            text="We should ship on Tuesday.",
            sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        )
    )
    second = chat_message_to_document(
        ChatMessage(
            chat_id=-1001234567890,
            message_id=43,
            user_id=8,
            session_id="chat:-1001234567890",
            author_name="Bob",
            username=None,
            text="Bob prefers Friday instead.",
            sent_at=datetime(2024, 1, 15, 12, 31, tzinfo=timezone.utc),
        )
    )
    return [first, second]


def _render_prompt(*, documents: list[Document], instruction: str) -> list[str]:
    builder = ChatPromptBuilder(
        template=list(SUMMARIZATION_PROMPT_TEMPLATE),
        required_variables=["documents", "instruction"],
    )
    rendered = builder.run(documents=documents, instruction=instruction)["prompt"]
    return [message.text or "" for message in rendered]


def test_rendered_prompt_contains_instruction_and_document_content() -> None:
    documents = _discussion_documents()
    rendered = ChatPromptBuilder(
        template=list(SUMMARIZATION_PROMPT_TEMPLATE),
        required_variables=["documents", "instruction"],
    ).run(documents=documents, instruction="Подготовь резюме обсуждения")["prompt"]

    assert [message.role for message in rendered] == [ChatRole.SYSTEM, ChatRole.USER]
    user_text = rendered[1].text
    assert "Подготовь резюме обсуждения" in user_text
    # The whole message line reaches the model: timestamp, author and text.
    assert documents[0].content in user_text
    assert documents[1].content in user_text
    assert "[2024-01-15T12:30:00+00:00] Alice (@alice): We should ship on Tuesday." in user_text


def test_rendered_prompt_preserves_document_order() -> None:
    documents = _discussion_documents()
    user_text = _render_prompt(documents=documents, instruction="Summarize")[1]

    first_index = user_text.index(documents[0].content)
    second_index = user_text.index(documents[1].content)
    assert first_index < second_index


def test_system_prompt_contains_grounding_and_injection_guards() -> None:
    system_text = _render_prompt(documents=_discussion_documents(), instruction="Summarize")[0]

    assert "только переданный контекст" in system_text
    assert "prompt injection" in system_text
    assert "Не выдумывай" in system_text


def test_system_prompt_requires_decisions_and_actions_distinction() -> None:
    system_text = _render_prompt(documents=_discussion_documents(), instruction="Summarize")[0]

    assert "принятые решения" in system_text
    assert "предложения" in system_text
    assert "Следующие действия" in system_text


def test_system_prompt_requires_completeness_rules() -> None:
    system_text = _render_prompt(documents=_discussion_documents(), instruction="Summarize")[0]

    assert "Проанализируй каждое переданное сообщение" in system_text
    assert "поручения" in system_text
    assert "ответственных" in system_text
    assert "сроки" in system_text
    assert "резервные каналы связи" in system_text
    assert "Запрещено писать, что следующие действия отсутствуют" in system_text
    assert "внутреннюю проверку полноты" in system_text
    assert "Не добавляй новые даты" in system_text


def test_system_prompt_requires_ai_recommendation_label() -> None:
    system_text = _render_prompt(documents=_discussion_documents(), instruction="Summarize")[0]

    assert "рекомендацию AI" in system_text


def test_prompt_does_not_include_document_ids_or_scores() -> None:
    documents = _discussion_documents()
    scored_documents = [
        Document(
            id=documents[0].id,
            content=documents[0].content,
            meta=dict(documents[0].meta),
            score=0.99,
        ),
        Document(
            id=documents[1].id,
            content=documents[1].content,
            meta=dict(documents[1].meta),
            score=0.88,
        ),
    ]
    combined = "\n".join(_render_prompt(documents=scored_documents, instruction="Summarize"))

    assert documents[0].id not in combined
    assert documents[1].id not in combined
    assert "0.99" not in combined
    assert "0.88" not in combined


def test_repeated_render_does_not_retain_previous_input() -> None:
    documents = _discussion_documents()
    _render_prompt(documents=documents, instruction="First instruction")
    second_render = _render_prompt(
        documents=[documents[0]],
        instruction="Second instruction",
    )[1]

    assert "First instruction" not in second_render
    assert "Second instruction" in second_render
    assert documents[1].content not in second_render

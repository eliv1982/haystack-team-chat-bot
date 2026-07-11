# haystack-team-chat-bot

Telegram bot for a group chat that indexes messages, retrieves relevant context, and generates summaries using Haystack pipelines backed by Pinecone.

## Planned Haystack Pipelines

1. **Indexing** — embed and store chat messages in Pinecone.
2. **Query / Retrieval** — find relevant messages for a user question.
3. **Summarization** — generate a concise answer from retrieved context.

## Current Status

**Stage 1 foundation** is complete.

**Stage 2A indexing pipeline** is implemented with an offline-testable flow:

`ChatMessage -> Haystack Document -> OpenAIDocumentEmbedder -> DocumentWriter -> PineconeDocumentStore`

**Stage 2B live indexing smoke** completed: direct OpenAI embeddings verified with `text-embedding-3-small`; Pinecone preflight PASS; exact document verification PASS; embedding dimension 1536 verified; smoke-document cleanup PASS. Live smoke uses namespace `haystack-team-chat-homework`.

**Stage 3A offline query/retrieval pipeline** is implemented:

```text
Query
→ OpenAITextEmbedder
→ PineconeEmbeddingRetriever
→ chat_id + session_id filters
→ validated Documents
```

Retrieval is isolated by chat and session metadata.

**Stage 3B live filtered retrieval** completed: semantic retrieval verified; exact `chat_id + session_id` isolation verified; contradictory records from another session and another chat were excluded; exact cleanup completed.

**Stage 4A offline summarization pipeline** is implemented. The project now has three Haystack pipelines:

1. indexing;
2. query/retrieval;
3. summarization.

```text
SummarizationRequest
→ validated RetrievalService context
→ ChatPromptBuilder
→ OpenAIChatGenerator
→ SummarizationResult
```

The LLM receives only documents that passed chat/session validation. The prompt distinguishes facts, proposals, decisions, and action items.

**Stage 4B live grounded summarization** completed: one filtered retrieval and one LLM generation verified; final summary contained positions, decision and action items; foreign session/chat facts were excluded; exact cleanup completed. All three required pipelines are live-verified.

**Stage 5A offline Telegram foundation** is implemented:

```text
Telegram Message
→ telegram adapter
→ domain ChatMessage
→ thread-safe InMemorySessionStore
```

The TeleBot factory is implemented, but polling and handlers are not wired yet. Session state is in-memory only. No live Telegram test has been run yet.

**Stage 5B offline listening flow** is implemented:

```text
/start_listening
→ InMemorySessionStore

group text
→ Telegram adapter
→ IndexingService
→ message_count +1

/stop_listening
→ final session snapshot
```

Handlers `/start_listening`, ordinary group text capture, and `/stop_listening` are registered without polling. Ordinary messages are indexed before the session counter increments. Session state remains in-memory. The «Что думаешь?» summary handler is not implemented yet. No live Telegram test has been run yet.

Query, summarization delivery, and the summary command handler are not implemented yet.

The project uses the direct OpenAI API by default. `OPENAI_BASE_URL` is optional and only needed for a custom OpenAI-compatible proxy endpoint.

## Requirements

- Python 3.10+

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values locally. **Never commit `.env`** — it is listed in `.gitignore`.

## Reference

See `Pipeline example.ipynb` in the project root for the official Haystack pipeline structure used as a reference for later stages.

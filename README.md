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

Query, summarization, and Telegram handlers are not implemented yet.

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

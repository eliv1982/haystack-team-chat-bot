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

Retrieval, summarization, Telegram handlers, and live Pinecone/OpenAI verification are not implemented yet. No live API calls to OpenAI, Pinecone, or Telegram have been performed in this stage.

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

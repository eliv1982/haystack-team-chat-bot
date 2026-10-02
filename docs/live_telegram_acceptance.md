# Live Telegram Acceptance

A manual, reproducible acceptance procedure for the Telegram path against real Telegram, OpenAI and Pinecone. It is not part of CI. The bot's replies are in Russian; their opening words are quoted below.

## Evidence status

- **One run is recorded against the current implementation:** [Live run 2026-10-02](#live-run-2026-10-02-current-implementation), at commit `3f32b8cd08371c762bd2a2d53df0df1b054f8825`. It is a single run, one chat, nine messages.
- The [Historical run (2026-07-12)](#historical-run-2026-07-12) exercised an earlier version whose summary used semantic top-k retrieval. That path was replaced by whole-session loading with a completeness gate, so the historical run does not verify today's summary behavior.
- The repository holds **no screenshots, GIFs or logs** from any run. The 2026-10-02 bot log was inspected locally and is summarized in the record, not stored. Richer visual evidence is optional packaging work; strip personal data before adding any.
- Refusal paths that are impractical or inappropriate to trigger live (1,000+ messages, a store holding more documents than counted, embedding failure, an incomplete store after the bounded retry, polling errors) are covered by the offline suite only.

## Prerequisites

| Check | Expected |
| ----- | -------- |
| Repository state | clean tree; record `git rev-parse HEAD` |
| Offline verification | `python -m pytest -q`, `python -m ruff check .`, `python -m pip check` all pass |
| `.env` | all required variables set, pointing at a **test** index and namespace |
| Pinecone index | exists and is ready; dimension and metric match the settings |
| Telegram | a throwaway test supergroup; the bot is a group administrator (or privacy mode is off) |
| Participants | at least two accounts, so author attribution can be checked |
| Process | exactly one bot process for this token |
| Telemetry | optionally `HAYSTACK_TELEMETRY_ENABLED=False` in the shell, see the README |

## Procedure

Start `python bot.py`. Startup must log the Pinecone preflight and `Telegram polling starting` without an `ERROR` or traceback, and the command menu must appear in the group.

| # | Action | Expected |
| - | ------ | -------- |
| 1 | `/help` | static instruction listing all five commands |
| 2 | `/status`, then `/summary` (no session yet) | status: no active or completed sessions («Активной записи и завершенных сессий…»); summary: no recorded discussion («В этом чате пока нет записанного обсуждения») |
| 3 | `/start_listening` | announcement («Запись обсуждения начата…») saying only text messages are saved |
| 4 | `/start_listening` again | «Запись обсуждения уже идет.» |
| 5 | Two accounts send four text messages (scenario below) | no replies; each is recorded |
| 6 | Send a photo, a sticker, a command and «Подведи итог» | none of them is counted |
| 7 | `/status` | active; count equals the number of text messages sent in step 5 |
| 8 | `/summary` | structured summary (criteria below); a following `/status` shows the same count |
| 9 | `/stop_listening` | «Запись обсуждения остановлена. Сохранено сообщений: N.» with N from step 7 |
| 10 | `/summary` | summary of the completed session; `/status` shows the completed session |
| 11 | «Подведи итог» | same as `/summary` |
| 12 | Restart the bot, then `/status` and `/summary` | as step 2: the in-memory registry is gone; the documents remain in Pinecone |
| 13 | `Ctrl+C` | `Telegram polling stopped` and no traceback |

Example scenario for step 5 (any comparable content works):

```text
Анна:   Предлагаю провести запуск во вторник в 16:00.
Борис:  Лучше в среду в 10:00, останется день на проверку.
Анна:   Решили: запуск во вторник в 16:00.
Марта:  Подготовлю чек-лист к понедельнику до 12:00, резервный канал связи — email.
```

**Summary criteria (steps 8 and 10):**

- sections present: «Тема», «Ключевые позиции», «Решения», «Следующие действия», «Нерешенные вопросы», «Рекомендация AI»;
- both proposals appear as positions; the final decision is distinguished from them;
- the action item carries its owner, deadline and fallback contact channel;
- the AI recommendation is labeled as a recommendation;
- nothing is invented, and no internal terms (Pinecone, embeddings, IDs) appear.

**Expected log line** (the application's log lines contain no message text, prompts or credentials):

```text
Summary completed: message_count=N source_count=N session_state=active|completed
```

`message_count` and `source_count` must be equal. If a summary is requested within moments of the last message, you may instead see a reply starting «Итог не подготовлен…» together with `Session not fully visible yet, retrying` or `Session still incomplete, summary refused`. That is the completeness gate working as designed (Pinecone is eventually consistent). Repeat `/summary` after a few seconds; it must then succeed. A refusal that persists is a defect to investigate.

## Cleanup

The run leaves documents in your test namespace. Remove them as described under *Manual purge* in the [README](../README.md#data-handling-and-privacy).

## Record of a run

Copy this table into the notes of the run you perform.

| Field | Value |
| ----- | ----- |
| Date | |
| Commit (`git rev-parse HEAD`) | |
| Working tree clean | |
| Python / `pyTelegramBotAPI` / `haystack-ai` | |
| Chat type, participants, messages sent | |
| Steps 1–13 result | |
| Deviations and defects | |
| Evidence stored (screenshots, logs) | |

## Live run 2026-10-02 (current implementation)

The first run of this procedure against the whole-session summary path, on the repository exactly as committed (no product code was changed for the run).

**How to read the evidence.** Replies visible only in Telegram (help text, status and stop replies, summary text) were observed by the repository owner in the group and reported to the person running the verification, who has no view into the chat. Message counts, the loaded source set, ordering and isolation were confirmed independently from the bot log and a read-only Pinecone check. The summary text is stored nowhere, so judgments about its content rest on the owner's reading; summary quality is not evaluated automatically. Evidence below is tagged **owner** (reported from Telegram), **log** (bot log) or **Pinecone** (read-only check).

| Field | Value |
| ----- | ----- |
| Date | 2026-10-02 |
| Commit (`git rev-parse HEAD`) | `3f32b8cd08371c762bd2a2d53df0df1b054f8825`, equal to `main` and `origin/main` |
| Working tree clean | yes, before and after the live run (this record was written afterwards) |
| Offline suite at this commit | 720 passed (run locally after the live run) |
| Platform | Windows 11, Python 3.12.10 |
| Libraries | pyTelegramBotAPI 4.34.0, haystack-ai 2.31.0, pinecone-haystack 6.2.0, pinecone 9.1.0, openai 2.45.0 |
| Models | chat `gpt-4o-mini`, embeddings `text-embedding-3-small`, direct OpenAI API (no `OPENAI_BASE_URL`) |
| Pinecone | existing index, ready, dimension 1536 / metric cosine, matching the settings; namespace `haystack-team-chat-homework` (the legacy default). The index also holds other applications' namespaces; their contents were not read or written (only their vector counts appeared in index statistics). |
| Telemetry | Haystack telemetry disabled in the launching shell |
| Telegram | the same test supergroup as the 2026-07-12 run (confirmed from the stored `chat_id`); two participant accounts (two distinct stored `user_id` values in each session); bot privacy mode on (`can_read_all_group_messages` false); ordinary messages were received, which with privacy mode on implies the bot was a group administrator (not checked separately); no webhook set and no pending updates before start |
| Process | one bot process (venv launcher plus its interpreter); no other Python process on the machine running the bot; no `409 Conflict` in the log (a polling process elsewhere cannot be ruled out except by that absence) |
| Messages sent | session 1: 6 text messages; session 2: 3 text messages; plus a sticker, commands and the summary alias |
| Evidence stored | none in the repository |

### Live smoke scripts

Run first, from the repository root, with the commands in the README. Each script deletes its own synthetic documents and confirms the deletion. "Attempts" are polls two seconds apart.

| Script | Result | Visibility attempts | Cleanup |
| ------ | ------ | ------------------- | ------- |
| `scripts/smoke_test_indexing.py` | pass | 1 | confirmed (2 polls) |
| `scripts/smoke_test_retrieval.py` | pass | 2: one retry for index lag | confirmed (2 polls) |
| `scripts/smoke_test_summarization.py` | pass | 2: one retry for index lag | confirmed (2 polls) |

The summarization smoke's synthetic corpus includes a foreign session in the same chat and a document in another chat; the source set was exactly the four target documents. The configured namespace held 8 vectors before the three scripts and 8 after, so none were left behind.

**Two different things were verified, and neither stands in for the other.**

- **Retrieval smoke** (`smoke_test_retrieval.py`) probes a *semantic top-k retrieval capability* that is not part of the bot's runtime. Its pass says nothing about what the Telegram `/summary` does.
- **Whole-session Telegram summary runtime** is what the Telegram scenarios below exercised: a filter-only load of the complete session, an exact-count completeness gate, and a chronological sort. In the log, every `/summary` shows a Pinecone filter query (`top_k=1000`, the filter-query cap, not `RETRIEVAL_TOP_K`) followed directly by the chat-completion call. **No embedding request occurs for `/summary`**; embedding requests appear only when a message is recorded.

### Telegram scenarios

Session 1 text, sent in this order by the two accounts (chronology test: message 3 supersedes message 1, and message 4 supersedes message 3):

```text
1. Предлагаю провести релиз в понедельник 13 октября в 15:00.
2. Не согласен: в понедельник у нас нет QA. Давайте во вторник 14 октября в 11:00.
3. Хорошо, договорились: релиз во вторник 14 октября в 11:00.
4. Стоп, во вторник сервер на обслуживании. Итоговое решение: релиз переносим на четверг 16 октября в 12:00, вторник отменяется.
5. Борис подготовит релиз-ноты до среды 15 октября 18:00.
6. Анна уведомит клиентов о новой дате до среды 15 октября 12:00, канал — email.
```

Session 2 text (a deliberately unrelated discussion):

```text
1. Давайте закажем обед на пятницу: пицца или суши?
2. Я за суши. Закажу до 12:00 в пятницу, оплата картой.
3. Решено: суши, заказывает Ирина до 12:00 в пятницу.
```

| Procedure step | Result | Evidence |
| -------------- | ------ | -------- |
| Startup | clean: settings, Pinecone preflight, pipelines, `Telegram polling starting`; no `ERROR` or traceback | log |
| Command menu | the five commands are registered for group chats | read via `getMyCommands`; menu appearance in the group was not separately reported |
| 1 `/help` | pass | owner |
| 2 `/status`, `/summary` with no session | pass | owner |
| 3 `/start_listening` | pass | owner |
| 4 repeated `/start_listening` | pass; the existing session was preserved | owner |
| 5 six text messages | six recorded: six embed-and-write cycles in the log, six stored documents | log, Pinecone |
| 6 non-recorded input | a sticker was ignored; `/help` was not counted; «Подведи итог» produced a summary and was not counted. **No photo was sent.** | owner; Pinecone shows no command or alias text stored |
| 7 `/status` | exactly 6 | owner |
| 8 `/summary` (active) | pass; `/status` afterwards still 6. The log shows two active-session summaries (the alias and the explicit `/summary`), each `Summary completed: message_count=6 source_count=6 session_state=active` | owner, log |
| 9 `/stop_listening` | pass; 6 saved | owner |
| 10 `/summary` after stop | pass; logged `message_count=6 source_count=6 session_state=completed`; `/status` showed no active session and a completed one | owner, log |
| 11 alias | exercised once, during the active session (above); not repeated after the stop | owner, log |
| Session 2 `/start_listening`, three messages | pass; three recorded | owner, log, Pinecone |
| Session 2 `/status` | exactly 3 | owner |
| Session 2 `/summary` | pass; logged `message_count=3 source_count=3 session_state=active` | owner, log |
| Session 2 `/stop_listening` | pass; 3 saved | owner |
| 12 restart bot, registry lost | **not run** | n/a |
| 13 `Ctrl+C` clean shutdown | **not verified.** A console Ctrl+C sent from another process had no effect on the Windows process, so the bot was terminated forcibly and `Telegram polling stopped` was not observed. | log |

### Grounding and isolation

- **Chronology (owner).** The session 1 summary showed the 13 October proposal, the move to 14 October, the maintenance conflict, and the final decision as **16 October at 12:00**. The explicit `/summary` agreed with the alias summary. The summary requested after the stop used the completed session and showed the same final decision.
- **Owners and deadlines (owner).** Boris: release notes by 15 October 18:00. Anna: client notification by 15 October 12:00 by email. Both appear in the participants' messages above.
- **Second session (owner).** The summary covered only the Friday lunch: decision sushi, owner Irina, deadline Friday 12:00, card payment. None of the session 1 content appeared: no release, Boris, Anna, 16 October or email.
- **Stored data (Pinecone, read-only, metadata and set-membership only).**
  - Session 1 holds 6 documents and session 2 holds 3. Each has one `chat_id` and one `session_id`, unique message and document IDs, `source` equal to `telegram`, and two distinct authors.
  - Each expected message text is present exactly once. Sorted by `(sent_at, message_id)`, both sessions come out in the order sent.
  - Neither session contains text from the other, and no command or summary-alias text was stored.
  - The summary-path query (hard `chat_id` + `session_id` filter) returned exactly 6 and 3 documents, the same IDs as the unfiltered set, matching `source_count` in the log.
  - The two sessions share a chat with each other and with the two July sessions, whose 8 documents carry the same `chat_id`. Those 8 were excluded from both summaries and are unchanged. **A second chat was not used**: cross-chat exclusion on the production summary path was exercised live only by the summarization smoke's synthetic corpus, and offline.
- **Observation (owner).** The «Рекомендация AI» section adds advice the participants did not state. It is labeled as an AI recommendation and kept apart from the decisions and actions. This is the intended design, but a reader should not take that section as discussion content.
- **Not separately reported for this run:** the presence of every section heading, and the absence of internal terms (Pinecone, embeddings, IDs), in the summary text. The summarization smoke checks both automatically on its synthetic corpus.

### Logs and privacy

The whole bot log (67 lines, stdout and stderr, third-party libraries included) was scanned. Credentials were searched by exact value taken from `.env`, and all patterns were also checked by shape.

| Check | Result |
| ----- | ------ |
| Telegram bot token (full value, secret part, bot-ID part) | absent |
| OpenAI and Pinecone API keys | absent |
| Key-shaped strings, `Authorization` headers, `api.telegram.org/bot` URLs | none |
| Message content (distinctive words from both sessions) | none; the log contains no Cyrillic characters at all |
| `Traceback`, `ERROR`, `WARNING`, `CRITICAL`, `409 Conflict` | none |
| Chat or user identifiers (`chat_id=`, `user_id`) | none |

Third-party libraries log request lines for OpenAI and Pinecone (`httpx`), the Pinecone data-plane hostname, and Haystack component progress. None of these contain credentials or message text in this run. This is a single run's observation, not an audit of those libraries.

### Pinecone visibility on the Telegram path

No `Session not fully visible yet` or `Session still incomplete` line appeared, so **the bot's own completeness-gate retry and refusal paths were not exercised live**. The first `/summary` of each session came a little under three minutes after the last recorded message (about 2 min 45 s in session 1 and 2 min 55 s in session 2, by log timestamps), well after index lag. Index lag itself was observed live only in the retrieval and summarization smokes, which each needed a second poll; those scripts use their own polling loops, not the bot's gate. The gate's retry and refusal behavior is covered offline.

### Not exercised live

Covered by the deterministic offline suite or by automated tests only, and not claimed as live-verified:

- Telegram-safe splitting of replies over 4096 characters. Every summary in this run was far shorter, and no expensive generation was forced to trigger it.
- A message from an anonymous group admin or another unsupported sender during an active session, and a bot-authored message.
- Embedding failure, a startup exception carrying the bot token, an incomplete store after the bounded retry, a store holding more documents than counted, refusal at 1,000 or more messages, and polling-error behavior. No credentials or services were deliberately disrupted.
- Procedure step 12 (restart and loss of the in-memory registry) and step 13 (clean `Ctrl+C` shutdown).
- A photo as the non-text input (a sticker was used).
- The BotFather privacy-mode-off configuration (the bot was run with privacy mode on).

### Deviations, defects and cleanup

- **Defects:** none found; no product code was changed.
- **Deviations from the template:** no photo; a sticker only; one chat; the Pinecone index is shared with other applications rather than dedicated to testing, so only this project's namespace was touched; steps 12 and 13 as above.
- **Cleanup:** the smoke documents were deleted by the scripts. After the run the namespace held 17 documents: the nine from this run (session IDs `telegram-session-a16f31aa-8b83-40af-9472-37a7284af0a0` with 6 and `telegram-session-2ef05be6-bc2c-4694-ab50-1d8a70bcce54` with 3) and the 8 from 2026-07-12.
  - **Purged the same day.** The nine documents were removed by exact `(chat_id, session_id)` filter, one session at a time, using the README's single-session purge (`delete_by_filter` with `build_session_filter`) after confirming 17 in the namespace and exactly 6 and 3 in the two sessions. 6 and 3 were deleted; afterwards neither session ID returned any document and the namespace held 8.
  - **Historical data untouched.** The 8 July documents remained: their IDs, metadata, content hashes and embedding lengths were identical before and after.
  - **No sibling namespace was modified:** only their vector counts (7 and 15) were read from the index statistics, and they were the same afterwards.
  - **Scope of the verification.** This confirms the single-session purge procedure live, once. It is not a retention or deletion feature, and the chat-wide filter and `delete_all_documents()` were not run. One unfiltered listing query returned 16 of 17 documents while the index statistics and the ID list showed 17 (the next returned 17), so verify counts with the index statistics or the exact session filter rather than a single unfiltered listing.

## Historical run (2026-07-12)

Recorded here for completeness, unchanged apart from the two notes marked below. Its scope is limited, and it must not be read as acceptance of the current build; the current build's run is [above](#live-run-2026-10-02-current-implementation).

| Field | Value |
| ----- | ----- |
| Date | 2026-07-12 |
| Recorded baseline commit | `40cc9ac` |
| Python / pyTelegramBotAPI / haystack-ai | 3.12.10 / 4.34.0 / 2.31.0 |
| Pinecone dimension / metric | 1536 / cosine |
| Scenario | one test supergroup, a four-message discussion |

Limitations of this record:

- **Superseded summary path.** At that time `/summary` retrieved context by semantic similarity. The whole-session path, the completeness gate and the refusal behavior were added afterwards (2026-10-02) and have not been run live in Telegram. *[Note added 2026-10-02: they were first run live on that date; see the [current run](#live-run-2026-10-02-current-implementation).]*
- **Unclear tested tree.** The recorded baseline `40cc9ac` predates most of what the run verified: at that commit only `/start_listening` and `/stop_listening` were handled, with no `/summary`, phrase aliases, `/status`, `/help`, command menu, latest-completed lookup or stricter prompt completeness rules. Those were committed in `0839e4f` together with this document, and the exact tree that was run was not recorded.
- **Small scale.** One run, four messages, one chat.

What the run reported:

1. A first launch started but received no updates because of an operational configuration problem.
2. Telegram `409 Conflict` was traced to a second polling process on the same bot token. This is an operational issue, not a code defect.
3. A first functional run covered start, duplicate start, four indexed messages, a grounded summary and a stop count of 4. It exposed three gaps: no summary after stop, an action item missing from the summary, and limited UX.
4. Targeted repairs followed: latest-completed session lookup, `/summary`, phrase aliases, `/status`, `/help`, the command menu and stricter prompt completeness rules.
5. A final rerun on the repaired build passed: command menu, start and duplicate start, `/status`, a four-message recording that stayed at 4 after `/summary`, summaries of the active and the latest completed session, action-item owner, deadline and contact channel in the summary, the phrase alias, `/help`, and a clean shutdown.

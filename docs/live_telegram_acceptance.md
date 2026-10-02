# Live Telegram Acceptance

A manual, reproducible acceptance procedure for the Telegram path against real Telegram, OpenAI and Pinecone. It is not part of CI. The bot's replies are in Russian; their opening words are quoted below.

## Evidence status

- **No run is recorded against the current implementation.** The only recorded run (see [Historical run](#historical-run-2026-07-12)) exercised an earlier version whose summary used semantic top-k retrieval. That path was replaced by whole-session loading with a completeness gate, so the run does not verify today's summary behavior.
- The repository holds **no screenshots, GIFs or logs** from any run. Richer visual evidence is optional packaging work; strip personal data before adding any.
- Refusal paths that are impractical to trigger live (1,000+ messages, a store holding more documents than counted) are covered by the offline suite only.

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

## Historical run (2026-07-12)

Recorded here for completeness. Its scope is limited, and it must not be read as acceptance of the current build.

| Field | Value |
| ----- | ----- |
| Date | 2026-07-12 |
| Recorded baseline commit | `45890f2` |
| Python / pyTelegramBotAPI / haystack-ai | 3.12.10 / 4.34.0 / 2.31.0 |
| Pinecone dimension / metric | 1536 / cosine |
| Scenario | one test supergroup, a four-message discussion |

Limitations of this record:

- **Superseded summary path.** At that time `/summary` retrieved context by semantic similarity. The whole-session path, the completeness gate and the refusal behavior were added afterwards (2026-10-02) and have not been run live in Telegram.
- **Unclear tested tree.** The recorded baseline `45890f2` predates most of what the run verified: at that commit only `/start_listening` and `/stop_listening` were handled, with no `/summary`, phrase aliases, `/status`, `/help`, command menu, latest-completed lookup or stricter prompt completeness rules. Those were committed in `2e4e396` together with this document, and the exact tree that was run was not recorded.
- **Small scale.** One run, four messages, one chat.

What the run reported:

1. A first launch started but received no updates because of an operational configuration problem.
2. Telegram `409 Conflict` was traced to a second polling process on the same bot token. This is an operational issue, not a code defect.
3. A first functional run covered start, duplicate start, four indexed messages, a grounded summary and a stop count of 4. It exposed three gaps: no summary after stop, an action item missing from the summary, and limited UX.
4. Targeted repairs followed: latest-completed session lookup, `/summary`, phrase aliases, `/status`, `/help`, the command menu and stricter prompt completeness rules.
5. A final rerun on the repaired build passed: command menu, start and duplicate start, `/status`, a four-message recording that stayed at 4 after `/summary`, summaries of the active and the latest completed session, action-item owner, deadline and contact channel in the summary, the phrase alias, `/help`, and a clean shutdown.

# Haystack Team Chat Bot

Telegram-бот для групповых рабочих обсуждений. Бот записывает текстовые сообщения участников в Pinecone, изолирует данные по чату и сессии и формирует структурированный итог через три Haystack pipeline.

## Возможности

- работа в Telegram group/supergroup;
- command menu с командами записи, итога, статуса и справки;
- запуск и остановка записи обсуждения;
- прием только текстовых сообщений участников;
- сохранение metadata автора;
- индексирование сообщений в Pinecone через indexing pipeline;
- active session и latest completed session на каждый чат;
- summary активной сессии;
- summary последней завершенной сессии после `/stop_listening`;
- natural-language aliases для запроса итога;
- команды `/status` и `/help`;
- source isolation по `chat_id + session_id`;
- отражение в summary решений, позиций, action items, ответственных, сроков, каналов связи и нерешенных вопросов;
- AI recommendation, явно отделенная от фактов обсуждения;
- thread-safe in-memory session store;
- production entry point `bot.py`;
- offline tests и live Telegram acceptance.

## Команды

| Команда            | Назначение                                              |
| ------------------ | ------------------------------------------------------- |
| `/start_listening` | Начать запись обсуждения                                |
| `/summary`         | Подвести итог активной или последней завершенной сессии |
| `/stop_listening`  | Остановить запись                                       |
| `/status`          | Показать состояние записи и счетчик сообщений           |
| `/help`            | Показать инструкцию                                     |

Natural-language aliases:

- `Подведи итог`
- `Подведи итог обсуждения`

Compatibility alias:

- `Что думаешь?`

Compatibility alias сохранен для соответствия учебному сценарию. Основной интерфейс для запроса итога — команда `/summary`.

## Что сохраняется

Сохраняются:

- только Telegram messages с `content_type=text`;
- только в group/supergroup;
- только во время active listening session;
- только после успешной индексации.

Не сохраняются:

- Telegram-команды;
- summary-команда `/summary`;
- summary aliases;
- фото;
- файлы;
- голосовые сообщения;
- видео;
- стикеры;
- опросы;
- service messages.

Команда `/stop_listening` закрывает сессию в in-memory registry, но не удаляет ее документы из Pinecone.

## Как работает бот

### Запись

```text
Telegram group text
→ Telegram adapter
→ domain ChatMessage
→ indexing pipeline
→ Pinecone
→ message_count +1
```

### Итог

```text
/summary или supported phrase
→ active session или latest completed session
→ retrieval pipeline
→ chat_id + session_id validation
→ summarization pipeline
→ Telegram reply
```

## Haystack pipelines

В проекте реализованы три Haystack pipeline.

### 1. Indexing pipeline

```text
ChatMessage
→ OpenAIDocumentEmbedder
→ DocumentWriter
→ PineconeDocumentStore
```

Компоненты соединены как `document_embedder.documents → writer.documents`. Политика записи — `DuplicatePolicy.OVERWRITE`.

### 2. Retrieval pipeline

```text
Query text
→ OpenAITextEmbedder
→ PineconeEmbeddingRetriever
→ chat_id + session_id filters
→ validated Documents
```

Retrieval service дополнительно проверяет документы: metadata, score, дубликаты и соответствие запрошенному chat/session.

### 3. Summarization pipeline

```text
Validated documents + instruction
→ ChatPromptBuilder
→ OpenAIChatGenerator
→ SummarizationResult
```

Prompt включает grounding rules, anti-injection, различение proposals/decisions/actions, completeness checks и один LLM call без fallback.

## Архитектура

| Модуль | Назначение |
| ------ | ---------- |
| `config.py` | загрузка и валидация Settings из environment |
| `models.py` | domain models и request/result types |
| `documents.py` | преобразование ChatMessage в Haystack Document |
| `document_store.py` | factory PineconeDocumentStore |
| `pinecone_preflight.py` | проверка существующего Pinecone index |
| `retrieval_filters.py` | filters по chat/session metadata |
| `summarization_prompt.py` | prompt template для summarization |
| `pipelines.py` | factories трех Haystack pipeline |
| `indexing_service.py` | запуск indexing pipeline |
| `retrieval_service.py` | запуск retrieval pipeline и validation |
| `summarization_service.py` | retrieval + summarization orchestration |
| `session_store.py` | thread-safe in-memory session registry |
| `telegram_adapter.py` | Telegram Message → ChatMessage, trigger detection |
| `telegram_application.py` | listening flow и status |
| `telegram_summary_application.py` | summary resolution и orchestration |
| `telegram_handlers.py` | регистрация Telegram handlers |
| `telegram_commands.py` | command menu для group/supergroup |
| `telegram_bot.py` | TeleBot factory |
| `runtime.py` | production assembly, menu setup, polling boundary |
| `bot.py` | production entry point |

```text
docs/
scripts/
tests/
Pipeline example.ipynb
.env.example
requirements.txt
requirements-dev.txt
pyproject.toml
.github/workflows/ci.yml
README.md
bot.py
runtime.py
pipelines.py
...
```

## Технологии

| Компонент | Версия |
| --------- | ------ |
| Python | 3.12 (проект проверен на 3.12.10; минимум 3.10+) |
| Haystack | haystack-ai 2.31.0 |
| Pinecone integration | pinecone-haystack 6.2.0 |
| Pinecone SDK | pinecone 9.1.0 |
| OpenAI API | через Haystack OpenAI components |
| PyTelegramBotAPI | 4.34.0 |
| python-dotenv | 1.1.0 |
| pytest | 8.4.1 |
| Ruff | 0.12.4 |

## Требования

- Python 3.10 или новее;
- существующий Pinecone index с dimension и metric, совместимыми с настройками;
- Telegram bot token;
- OpenAI API key.

## Установка

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Заполните `.env` локально. Не коммитьте `.env` в Git.

## Переменные окружения

| Переменная | Обязательность | Назначение |
| ---------- | -------------- | ---------- |
| `TELEGRAM_BOT_TOKEN` | обязательна | token Telegram Bot API |
| `OPENAI_API_KEY` | обязательна | ключ OpenAI API |
| `OPENAI_BASE_URL` | опциональна | custom OpenAI-compatible endpoint; для прямого OpenAI API не нужна |
| `OPENAI_MODEL` | обязательна | chat model для summarization |
| `EMBEDDING_MODEL` | обязательна | embedding model для indexing/retrieval |
| `PINECONE_API_KEY` | обязательна | ключ Pinecone |
| `PINECONE_INDEX_NAME` | обязательна | имя существующего index |
| `PINECONE_NAMESPACE` | опциональна | namespace; default `haystack-team-chat-homework` |
| `PINECONE_DIMENSION` | опциональна | dimension index; default `1536` |
| `PINECONE_METRIC` | опциональна | metric index; default `cosine` |
| `RETRIEVAL_TOP_K` | опциональна | top_k retrieval; default `50` |

`.env` — локальный файл с secrets. `.env` игнорируется Git. `.env.example` содержит только placeholders без secrets.

## Подготовка Telegram

1. Создайте тестовую или рабочую group/supergroup.
2. Добавьте бота в группу.
3. Назначьте бота администратором, чтобы он получал обычные сообщения группы.
4. Откройте command menu и проверьте команды бота.
5. Не публикуйте bot token.

## Запуск

```powershell
.\.venv\Scripts\python.exe bot.py
```

Startup sequence:

```text
Settings
→ existing Pinecone index preflight
→ DocumentStore
→ three pipelines
→ services
→ command menu
→ infinity polling
```

Важно:

- используется существующий Pinecone index; проект не создает index автоматически;
- при старте применяется `skip_pending=True`;
- одновременно допустим только один polling process на один bot token;
- ошибка Telegram `409 Conflict` означает второй одновременно работающий polling process, а не дефект pipeline.

## Использование

```text
/start_listening
→ сообщения участников
→ /status
→ /summary
→ /stop_listening
→ /summary
```

Summary trigger не увеличивает `message_count` и не индексируется. После `/stop_listening` итог последней завершенной сессии доступен через `/summary`.

## Тестирование

pytest и ruff вынесены из runtime-зависимостей в `requirements-dev.txt` (он включает `requirements.txt`):

```powershell
python -m pip install -r requirements-dev.txt
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
```

Работает и просто `pytest -q` (настройка в `pyproject.toml`).

Текущий результат offline suite: **720 passed**.

Suite полностью offline: `tests/conftest.py` запрещает соединения с любыми хостами, кроме loopback, и отключает Haystack telemetry (`HAYSTACK_TELEMETRY_ENABLED=False`). GitHub Actions (`.github/workflows/ci.yml`) запускает ruff, pytest и `pip check` на Python 3.10 и 3.12 без секретов и без обращений к OpenAI, Pinecone и Telegram.

Покрытие включает:

- offline unit/integration tests для adapter, session store, services, handlers, runtime и pipelines;
- live indexing smoke (`scripts/smoke_test_indexing.py`);
- live filtered retrieval smoke (`scripts/smoke_test_retrieval.py`);
- live grounded summarization smoke (`scripts/smoke_test_summarization.py`);
- live Telegram acceptance в реальной group/supergroup.

Smoke scripts запускаются вручную и не являются частью обычного CI entry point.

## Live acceptance

В реальной Telegram-группе подтверждены:

- command menu;
- start и duplicate start;
- `/status`;
- сценарий записи из четырех сообщений;
- count остался равен 4 после summary;
- summary активной сессии;
- stop count = 4;
- summary latest completed session после stop;
- action items, owner, deadline и contact channel в итоге;
- safe shutdown.

Подробнее: [Отчет о live-приемке](docs/live_telegram_acceptance.md).

Скриншоты не хранятся в репозитории и прикладываются к учебной сдаче отдельно.

## Ограничения текущей версии

- сохраняются только text messages;
- один active session на chat;
- одна latest completed session на chat;
- session registry in-memory;
- registry теряется после restart процесса;
- документы при этом остаются в Pinecone;
- нет выбора произвольной исторической session;
- нет persistent session database;
- нет автоматического удаления старых документов;
- используется long polling, не webhook;
- нет ingestion voice/files;
- качество summary зависит от retrieval и LLM.

## Дальнейшее масштабирование

Текущая версия — учебный MVP, который уже покрывает полный цикл записи, индексации и summary в Telegram-группе. Следующий этап развития — превращение этого MVP в полноценного командного AI-ассистента для рабочих обсуждений.

**Текущая точка:**

- реестр сессий хранится in-memory (`InMemorySessionStore`);
- для каждого чата доступна одна active session и одна latest completed session;
- после перезапуска процесса связь с сессиями теряется;
- документы обсуждений при этом остаются в Pinecone;
- выбор произвольной прошлой сессии пока не реализован.

### 1. Постоянное хранение сессий

Переход от `InMemorySessionStore` к persistent session registry:

- **SQLite** — для локального MVP и быстрой итерации;
- **PostgreSQL** — для production и многопроцессного runtime.

Планируемые поля реестра:

- `chat_id`, `session_id`, статус;
- название встречи;
- время начала и завершения;
- инициатор;
- количество сообщений;
- даты создания и обновления.

Дополнительно: восстановление active и completed sessions после перезапуска и транзакционное согласование session registry с документами в Pinecone.

### 2. История и выбор сессий

Будущий пользовательский сценарий:

- команда `/sessions`;
- список последних обсуждений с пагинацией;
- inline-кнопки Telegram для быстрого выбора;
- выбор сессии по дате или названию;
- `/summary <session>` для повторного получения итога;
- фильтрация по диапазону дат;
- закрепление важных встреч;
- архивирование сессий.

Именно это расширение позволит получать summary не только текущей или последней завершенной сессии, но и **любого выбранного обсуждения**.

### 3. Названия и управление встречами

- `/start_listening Название встречи`;
- автоматическое название по первым сообщениям;
- переименование сессии;
- теги и категории;
- привязка к проекту, команде или клиенту;
- поддержка нескольких параллельных Telegram topics внутри одной группы.

### 4. Structured meeting intelligence

Переход от текстового summary к структурированному результату:

- тема, участники и позиции;
- принятые решения;
- action items, ответственные, сроки;
- нерешенные вопросы;
- риски, блокеры, зависимости;
- резервные сценарии;
- ссылки и каналы связи.

Планируется Pydantic/structured output с хранением результата отдельно от исходных сообщений. Перспектива: журнал решений, реестр поручений, контроль просроченных задач, сравнение первоначальных договоренностей с последующими изменениями.

### 5. Поиск и вопросы по истории

- Q&A по одной выбранной сессии;
- поиск по всем обсуждениям конкретного чата;
- cross-session RAG;
- вопросы вида: «Когда мы приняли это решение?», «Кто отвечал за задачу?», «Какие сроки менялись?», «В каких встречах обсуждался этот риск?»;
- сравнение решений между сессиями;
- выявление противоречий;
- временная линия решений и поручений.

### 6. Экспорт и интеграции

Экспорт итогов (планируется, **не реализовано**):

- Markdown, DOCX, PDF;
- email, Slack, Google Docs, Notion;
- Jira, Trello;
- корпоративные базы знаний.

Дополнительно: автоматическая отправка протокола после завершения встречи, создание задач из action items, синхронизация сроков с календарем, уведомления ответственным.

### 7. Поддержка дополнительных форматов

Будущая обработка контента:

- голосовые сообщения с transcription;
- аудиозаписи встреч;
- документы, фотографии и подписи;
- ссылки;
- edited и deleted Telegram messages;
- replies и message threads;
- Telegram topics;
- вложения, привязанные к конкретной сессии.

Для каждого формата потребуется отдельный ingestion и validation pipeline.

### 8. Управление данными и приватность

- retention policy и срок хранения сообщений/embeddings;
- `/delete_session`, `/delete_history`;
- удаление данных конкретного чата;
- role-based access;
- разрешение на summary только администраторам или участникам;
- audit log, шифрование, data minimization;
- согласие участников на запись обсуждения;
- исключение чувствительных сообщений;
- поддержка корпоративных требований к хранению данных.

### 9. Улучшение пользовательского опыта

- автоматический summary при `/stop_listening` как opt-in;
- промежуточные summaries по расписанию;
- напоминания по action items;
- выбор формата: краткий итог, полный протокол, только решения, только поручения, риски и нерешенные вопросы;
- многоязычные summaries;
- настройка тона и детализации;
- feedback/rating;
- повторная генерация с уточненной инструкцией;
- inline-кнопки вместо необходимости помнить команды.

### 10. Production-масштабирование

Переход от учебного long polling к production runtime:

- webhook mode;
- Docker, VPS или облачное развертывание;
- PostgreSQL, очереди задач, background workers;
- rate limits, retries, dead-letter queue;
- health endpoint, metrics, structured logs, tracing, alerting;
- резервное копирование и горизонтальное масштабирование;
- разделение Telegram ingestion и AI processing;
- контроль расходов OpenAI и Pinecone;
- кэширование повторных summary;
- нагрузочное тестирование.

### 11. Возможное развитие архитектуры Haystack

- отдельные pipeline для разных типов задач: summary, action-item extraction, decision extraction, Q&A, document ingestion;
- routing между pipeline;
- evaluation datasets и retrieval quality metrics;
- prompt/version management;
- observability Haystack components;
- A/B-тестирование prompts и retrieval settings.

### 12. Этапы развития

```text
Этап 1 — persistent session registry
Этап 2 — /sessions и выбор исторической сессии
Этап 3 — structured decisions и action items
Этап 4 — cross-session search и Q&A
Этап 5 — exports и внешние интеграции
Этап 6 — rich content и voice
Этап 7 — privacy controls и production deployment
```

Подробный технический план: [docs/project_roadmap.md](docs/project_roadmap.md).

## Безопасность

- `.env` игнорируется Git;
- secrets не коммитятся;
- сообщения участников передаются в prompt как недоверенные данные;
- prompt содержит anti-injection rules;
- retrieval и summarization изолированы по exact chat/session;
- полные messages, prompts и secrets не логируются;
- screenshots с персональными данными не хранятся в public repository.

## Reference

- `Pipeline example.ipynb` — учебный reference по структуре Haystack pipeline;
- официальная документация Haystack: [https://docs.haystack.deepset.ai/](https://docs.haystack.deepset.ai/);
- notebook не является runtime dependency.

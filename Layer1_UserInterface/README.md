# AI Property Triage — Layer 1 (Web UI)

Streamlit interface for an AI-powered real-estate triage platform.
Listing agents can submit a property for automated analysis (Tab 2) and
chat with an AI assistant about real-estate questions or their submitted
listings (Tab 1).

This folder is **Layer 1** in the full architecture:

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Layer 1 (us)   │    │   Layer 2    │    │     Layer 3     │    │   Layer 4    │
│  Streamlit UI   │───▶│  n8n flow    │───▶│ FastAPI on EC2  │───▶│ External LLM │
│  (this folder)  │    │  (25 nodes)  │    │ RAG / Image /…  │    │ Gemini / …   │
└─────────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
```

---

## What's in this folder

| File                  | Role                                                    |
|-----------------------|---------------------------------------------------------|
| `app.py`              | Streamlit entry point — UI, tabs, sidebar, chat loop    |
| `chat_client.py`      | Groq (Llama 3.1 8B) client + system prompt (V12)        |
| `sonar_client.py`     | Perplexity Sonar client for live web data               |
| `router.py`           | Keyword router: groq / perplexity / pinecone            |
| `pinecone_client.py`  | Vector store for all submitted listings                 |
| `db.py`               | SQLite layer for chat sessions + last-listing cache     |
| `chat_history.db`     | SQLite database (auto-created on first run)             |
| `style.css`           | Custom UI styling                                       |
| `PROMPT_LOG.md`       | Full prompt-engineering log (V1 → V12 + Sonar + RAG)    |

---

## Architecture (this layer)

```
                ┌─── Tab 1: AI Assistant ───┐
                │                           │
  user message ─┤        router.route()     │
                │           │ ▼             │
                │  ┌────────┼────────┐      │
                │  ▼        ▼        ▼      │
                │ Groq   Perplexity Pinecone│
                │  │        │     (vector)  │
                │  ▼        ▼        │      │
                │ stream answer      │      │
                │                    ▼      │
                │           top-k matches   │
                │           → Groq summary  │
                └───────────────────────────┘
                            ▲
                            │  history + last listing
                            │
                ┌─── SQLite (chat_history.db) ───┐
                │  chat_sessions / messages       │
                │  last_listing                   │
                └─────────────────────────────────┘

                ┌─── Tab 2: Submit Listing ──┐
                │                            │
  form payload ─▶ POST localhost:5678/…  ───▶ n8n workflow (Layer 2)
                │           │                │
                │           ▼                │
                │      report JSON           │
                │           │                │
                │           ├─▶ render UI    │
                │           ├─▶ db.save_last_listing()
                │           └─▶ pinecone_client.upsert_listing()
                └────────────────────────────┘
```

---

## Environment variables required

Set these in PowerShell (Windows) before running:

```powershell
$env:GROQ_API_KEY     = "gsk_..."          # https://console.groq.com
$env:PPLX_API_KEY     = "pplx_..."         # https://www.perplexity.ai/settings/api
$env:PINECONE_API_KEY = "pcsk_..."         # https://app.pinecone.io
$env:GEMINI_API_KEY   = "AIza..."          # https://aistudio.google.com/app/apikey
```

To persist them across sessions:

```powershell
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", "gsk_...", "User")
# repeat for the others
```

---

## Install + run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

You also need n8n running with the Property Triage workflow imported and
active on port `5678`. See `../Layer2_n8n/LAYER2_README.md` for the n8n
side.

---

## How the chat picks a backend (router)

`router.route(message)` inspects the user message and returns one of:

| Decision     | When                                                              | Used by                                |
|--------------|-------------------------------------------------------------------|----------------------------------------|
| `groq`       | Greetings, definitions, evergreen advice, refusals (default)      | Llama 3.1 8B via Groq                  |
| `perplexity` | "Current X", "today's rate", "USD to ILS", live market data       | Perplexity Sonar (live web search)     |
| `pinecone`   | "Show my listings", "that villa I uploaded", "compare my reports" | Pinecone retrieval → Groq summariser   |

The router is deterministic regex-based — no LLM call needed for routing.
This is the documented workaround for the 8B model's unreliable tool
calling (see `PROMPT_LOG.md`, iteration V8).

---

## How listings are stored

Two stores, two different jobs:

| Store            | What                                              | Why                                              |
|------------------|---------------------------------------------------|--------------------------------------------------|
| **SQLite**       | The single most recent listing (`last_listing`)   | So Tab 1 can answer "what do you think of my listing?" with specific details |
| **Pinecone**     | EVERY listing ever submitted, as 768-d embeddings | So Tab 1 can answer "find that office I uploaded last week" by semantic search |

After every successful n8n response, `app.py` writes to BOTH stores.
Pinecone metadata stores the analysis report (location, price, rooms,
size, features) — not the raw base64 images, because Pinecone metadata
has a ~40 KB cap.

---

## Notes

- The chat caps responses at 3 sentences (or up to 20 when answering a
  multi-listing retrieval question — see `enforce_response_limit` in
  `chat_client.py`).
- Status checks (Groq / Perplexity / n8n / Pinecone) are cached for
  60 seconds via `@st.cache_data` so re-runs after every user message
  don't re-ping every service — that was the source of the perceived
  "page reload" lag.
- The chat history persists in `chat_history.db`. Delete that file to
  reset all chats.
- The `chat_history_OLD.json` file is the pre-SQLite chat store — kept
  for reference, no longer used by the app.
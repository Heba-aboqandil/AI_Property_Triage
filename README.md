# AI Property Triage Platform

> AI-powered real-estate triage — Streamlit + n8n (LangChain) + 4 EC2 microservices + Pinecone vector memory.

An end-to-end AI system that helps real-estate teams process incoming
property listings faster. An agent submits a free-text listing
(optionally with photos), and the platform validates it, extracts
structured fields, retrieves similar past listings via RAG, analyses
property images, generates a clean report, and routes the listing to the
right team — all in a few seconds.


## What the system does

The platform accepts a property listing submission through a Streamlit
web UI and processes it through a four-layer AI pipeline:

- **Input guardrails** — reject PII, abusive, and off-topic text before
  any LLM cost is incurred.
- **Structured field extraction** — Gemini LLM converts the free-text
  description into typed fields (location, price, rooms, size, features).
- **RAG enrichment** — Pinecone vector search retrieves similar past
  listings and the RAG service generates a market-insight paragraph.
- **Image analysis** — a fine-tuned ResNet-50 classifies each photo by
  room type and gives a 1–5 condition score.
- **Multi-step reasoning** (optional) — a LangGraph agent plans tool
  calls and synthesises deeper analyses when needed.
- **Output guardrails** — a second check on the AI-generated report
  catches hallucinations, price guarantees, and false legal claims.
- **Routing** — residential vs commercial decision; the final report is
  saved to Pinecone and shown back to the agent.

Beyond submissions, the **AI Assistant** tab is a real-estate chat
backed by three routes:

- Groq Llama 3.1 for general property knowledge,
- Perplexity Sonar for live market data (rates, exchange, prices),
- Pinecone for memory — “show me my listings” returns prior reports.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Layer 1 — Web UI  ·  Streamlit  ·  :8501                      │
│   Tab 1 — AI Assistant   (Groq / Perplexity / Pinecone router) │
│   Tab 2 — Submit Listing (form → n8n webhook)                  │
└─────────────────────────┬──────────────────────────────────────┘
                          │ POST /webhook/property-triage
┌─────────────────────────▼──────────────────────────────────────┐
│  Layer 2 — Orchestration  ·  n8n + LangChain  ·  :5678         │
│   Webhook → Guardrails In → Extract → AI Agent → Report →      │
│   Guardrails Out → Property Type Router → Notify Team           │
│                                                                 │
│   9 LangChain nodes: Information Extractor (Chain), AI Agent,  │
│   Final Report (Chain), 3 Tools (HTTP), 3 Gemini ChatModels.   │
└─────────────────────────┬──────────────────────────────────────┘
                          │ HTTP calls to four services
┌─────────────────────────▼──────────────────────────────────────┐
│  Layer 3 — AI Microservices  ·  FastAPI on AWS EC2             │
│   rag_service        :8001   Pinecone RAG + LLM insight        │
│   image_analyser     :8002   PyTorch ResNet-50                 │
│   guardrails_service :8003   NeMo Guardrails                   │
│   langgraph_agent    :8004   LangGraph StateGraph              │
└────────────────────────────────────────────────────────────────┘

         Layer 4 — External LLM APIs (called from layers 1 + 2)
         · Gemini (chat + embeddings)
         · Groq (Llama 3.1 8B Instant)
         · Perplexity Sonar
```

The submission lifecycle in one line:

> listing in → guardrails → extract → AI agent calls tools (RAG · Image
> · LangGraph) → report → output check → **routed**.

---

## Folder structure

```
AI_Property_Triage/
├── Layer1_UserInterface/        # Streamlit app 
│   ├── app.py                   # Two-tab UI, dark mode, status badges
│   ├── chat_client.py           # Groq client + V12 system prompt
│   ├── sonar_client.py          # Perplexity Sonar client
│   ├── router.py                # 3-route keyword router
│   ├── pinecone_client.py       # Vector store for submitted listings
│   ├── db.py                    # SQLite chat history
│   ├── style.css                # Luxe Light + gold accents
│   ├── PROMPT_LOG.md            # V1 → V12 iterations
│   ├── README.md
│   └── requirements.txt
│
├── Layer2_n8n/                  # Orchestration 
│   ├── AI_Property_Triage_v2.json   # 25-node workflow
│   └── LAYER2_README.md
│
├── final_project/layer3/        # FastAPI microservices 
│   ├── rag_service/             # /query  → Pinecone + LLM
│   ├── image_analyser/          # /analyse, /analyse/batch, /analyse/upload
│   ├── guardrails_service/      # /check/input, /check/output
│   ├── langgraph_agent/         # /agent/run
│   └── shared/schemas.py        # Pydantic models shared by all services
│
├── .gitignore
└── README.md                    # This file
```

---

## Tech stack

| Layer | Technology | What it does |
|---|---|---|
| **UI** | Streamlit | Two-tab interface (chat + submission), live status badges, dark/light themes |
| **Orchestration** | n8n + LangChain | 25-node visual workflow with Information Extractor, AI Agent, LLM Chain, 3 HTTP tools |
| **AI Agent** | LangChain Tools (MCP-style) | The agent dynamically picks tools: `rag_query`, `analyse_images`, `langgraph_agent` |
| **RAG** | Pinecone + Gemini embeddings | `gemini-embedding-001` at 768 dimensions; every submission is recallable |
| **LLMs** | Groq + Perplexity + Gemini | Llama 3.1 8B for chat, Sonar for live data, Gemini for n8n LangChain nodes |
| **Image AI** | PyTorch ResNet-50 | Fine-tuned for room-type classification + condition scoring |
| **Guardrails** | NeMo Guardrails | Input rail (reject) + Output rail (flag for human review) |
| **Multi-step** | LangGraph | StateGraph for deeper multi-step reasoning |
| **Services** | FastAPI on AWS EC2 | Four containerised microservices, public endpoints |
| **Data** | JSON + SQLite | JSON for every payload; SQLite for chat history |
| **Deployment** | Docker + GitHub | n8n in Docker; full repo on GitHub with `.env` protection |

---

## Port map

| Service | Port | Stack |
|---|---:|---|
| Streamlit UI | 8501 | Streamlit |
| n8n | 5678 | n8n |
| RAG | 8001 | FastAPI + Pinecone + LLM |
| Image Analyser | 8002 | FastAPI + PyTorch ResNet-50 |
| Guardrails | 8003 | FastAPI + NeMo Guardrails |
| LangGraph Agent | 8004 | FastAPI + LangGraph |

The four EC2 services run on `54.84.168.9` in the deployed environment.

---

## Quick start

> The fastest path to a working demo is **Layer 1 only** (chat). For the
> full submission pipeline, the EC2 services in Layer 3 must be up.

### Layer 1 — Streamlit UI (chat works standalone)

```bash
cd Layer1_UserInterface

# 1. Install
pip install -r requirements.txt

# 2. Configure secrets
cp ../.env.example ../.env
# edit ../.env and fill in:
#   GROQ_API_KEY=gsk_...
#   PPLX_API_KEY=pplx_...
#   PINECONE_API_KEY=pcsk_...
#   GEMINI_API_KEY=AIza...
#   N8N_WEBHOOK_URL=http://localhost:5678/webhook/property-triage

# 3. Run
streamlit run app.py
```

Open <http://localhost:8501>.

The **AI Assistant** tab works immediately. The **Submit Listing** tab
needs n8n (Layer 2) running.

### Layer 2 — n8n workflow

```bash
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n
```

Open <http://localhost:5678>:

1. Top-right menu → **Import from File** → pick
   `Layer2_n8n/AI_Property_Triage_v2.json`.
2. Open any **Gemini Chat Model** node → set the credential to a Google
   AI Studio API key (free tier: <https://aistudio.google.com/app/apikey>).
3. Toggle **Active** in the top-right.

The webhook is now live at:

```
http://localhost:5678/webhook/property-triage
```

### Layer 3 — EC2 microservices

The four Layer 3 services live in `final_project/layer3/` and are
deployed on AWS EC2 at `54.84.168.9` (ports 8001–8004). The Layer 2
workflow points to them by IP and port — no extra setup is needed on
the local machine.

To run Layer 3 locally instead, build the four containers under
`final_project/layer3/` and update the URL field in the corresponding
HTTP nodes in n8n.

---

## End-to-end test

With all three layers up:

1. Open the Streamlit UI → **Submit Listing** tab.
2. Paste an agent name and a property description (≥ 10 chars).
3. Optionally paste one or more public image URLs, or upload local
   `.jpg` / `.png` / `.webp` files.
4. Click **Submit**.

You should see the executions in n8n light up node by node, and
Streamlit render the structured report. The listing is then saved to
Pinecone and recallable from the chat.

### Sample listing

```
Beautiful 3-bedroom apartment in central Tel Aviv, 95 sqm, recently
renovated, sea view. Bright living room, modern kitchen with premium
appliances, two bathrooms. Near beach, restaurants, and public
transportation. Asking 4,200,000 NIS.
```

Sample image URL:

```
https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg
```

Expected:

- `routing_decision: residential`
- Location: Tel Aviv, 3 rooms, 95 sqm, ₪4,200,000
- Similar listings retrieved by RAG
- One image score with room type + condition score

---

## Layer-by-layer documentation

Each layer has its own README with deeper detail:

| Layer | Document |
|---|---|
| Layer 1 — Streamlit UI | [`Layer1_UserInterface/README.md`](Layer1_UserInterface/README.md) |
| Layer 1 — Prompt iterations (V1 → V12) | [`Layer1_UserInterface/PROMPT_LOG.md`](Layer1_UserInterface/PROMPT_LOG.md) |
| Layer 2 — n8n workflow | [`Layer2_n8n/LAYER2_README.md`](Layer2_n8n/LAYER2_README.md) |
| Layer 3 — EC2 services | `final_project/layer3/` (see each service's `README.md`) |

---

## Project status

| Component | Status |
|---|---|
| Streamlit UI — chat (3 routes) | Working |
| Streamlit UI — submission form | Working |
| n8n workflow — guardrails | Working |
| n8n workflow — extractor | Working |
| n8n workflow — AI Agent + tools | Working |
| n8n workflow — final report + routing | Working |
| Pinecone vector memory | Working |
| EC2 — RAG service | Working |
| EC2 — Image Analyser (URLs) | Working |
| EC2 — Image Analyser (uploads) | Working via Streamlit `/analyse/upload` |
| EC2 — Guardrails | Working |
| EC2 — LangGraph agent | Working |
| Team notification endpoints | Mock placeholders (replace with Slack / email in production) |

---

## Next steps

- Replace the mock notification endpoints with real Slack, email, or CRM
  integrations.
- Swap the EC2 local LLM for a managed endpoint to cut tool-call
  latency from minutes to seconds.
- Add native Arabic + Hebrew support to the chat assistant.
- Build an admin dashboard for the human-review queue and listing
  analytics.
- Move hard-coded EC2 IPs in the n8n workflow into environment variables
  so the JSON does not need editing when the IP changes.

---

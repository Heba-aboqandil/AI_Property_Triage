# Layer 2 — n8n Orchestration

The n8n workflow is the central orchestrator of the AI Property Triage
platform. It receives a property listing from the Streamlit UI, runs it
through a LangChain-based pipeline (input guardrails → information
extraction → AI Agent with three tools → final report → output guardrails
→ routing), and returns a structured JSON report.

---

## Folder contents

| File | Purpose |
|---|---|
| `AI_Property_Triage_v2.json` | The complete n8n workflow — import this into n8n to run it |
| `LAYER2_README.md` | This file |

The workflow connects to four FastAPI services deployed on AWS EC2 by
the Layer 3 teammate. No local mock services are needed; the workflow
points directly to the live EC2 endpoints.

---

## LangChain in n8n

The project guideline requires LangChain (or LangGraph) for the LLM
orchestration in n8n. The workflow uses n8n's built-in LangChain nodes
for every LLM operation. 9 of the 25 functional nodes are LangChain
components:

| Node | LangChain type | What it is |
|---|---|---|
| Node 4 — Information Extractor | `chainLlm` | LangChain LLMChain |
| Node 5 — AI Agent | `agent` | LangChain Agent |
| Node 6 — Final Report LLM Chain | `chainLlm` | LangChain LLMChain |
| rag_query | `toolHttpRequest` | LangChain Tool |
| analyse_images | `toolHttpRequest` | LangChain Tool |
| langgraph_agent | `toolHttpRequest` | LangChain Tool (calls a LangGraph agent on EC2) |
| 3 × Gemini Chat Model | `lmChatGoogleGemini` | LangChain ChatModel |

LangGraph itself runs on EC2 (Layer 3, the teammate's work). The
workflow calls it via the `langgraph_agent` tool.

---

## Webhook payload (from Streamlit)

The workflow receives this JSON shape on `POST /webhook/property-triage`:

```json
{
  "submission_id": "sub_YYYYMMDD_HHMMSS",
  "submitted_at": "2026-06-04T18:30:00.000",
  "agent_name": "Hiba Awadallah",
  "description": "Beautiful 3-bedroom apartment in central Tel Aviv...",
  "image_urls": ["https://example.com/photo1.jpg"],
  "images_base64": [],
  "precomputed_image_scores": []
}
```

Two image input modes are supported:

- **Pasted URLs** — `image_urls` is populated; the workflow calls the
  image analyser via the `analyse_images` tool.
- **Uploaded files** — Streamlit pre-analyses uploaded files against the
  EC2 `/analyse/upload` endpoint and ships the results in
  `precomputed_image_scores`. The workflow then skips `analyse_images`
  and uses the pre-computed scores directly.

---

## Workflow overview

```
Webhook
   │
   ▼
Guardrails Input Check ──── reject ──▶ Reject Response (422)
   │
   ▼ (pass)
Information Extractor  (LangChain LLMChain + Gemini)
   │
   ▼
Parse Extracted JSON  (code node)
   │
   ▼
AI Agent  (LangChain Agent + Gemini)
   ├── tool: rag_query        ──▶ EC2 RAG Service       (port 8001)
   ├── tool: analyse_images   ──▶ EC2 Image Analyser    (port 8002)
   └── tool: langgraph_agent  ──▶ EC2 LangGraph Agent   (port 8004)
   │
   ▼
Normalize Image Scores  (code node — Node 5b)
   │
   ▼
Final Report LLM Chain  (LangChain LLMChain + Gemini)
   │
   ▼
Parse Report JSON  (code node)
   │
   ▼
Guardrails Output Check ──── flag ──▶ Human Review Webhook ─▶ Respond Human Review
   │
   ▼ (pass)
Property Type Router  (switch on routing_decision)
   ├── residential ──▶ Notify Residential Team ──▶ 200 Response
   ├── commercial  ──▶ Notify Commercial Team  ──▶ 200 Response
   └── fallback    ──▶ Fallback Response
```

The full workflow has 25 functional nodes plus sticky-note annotations.

---

## Routing rules

| Property type | Routing decision |
|---|---|
| apartment, house, villa, condo, studio | `residential` |
| office, retail, warehouse, industrial | `commercial` |
| anything unclear | `fallback` |

The decision is captured in the final report (`routing_decision` field)
and visible to the agent who submitted the listing.

---

## EC2 service endpoints

The workflow calls the four Layer 3 services hosted on a single EC2
instance at `54.84.168.9`:

| Service | URL | Request schema |
|---|---|---|
| Guardrails Input | `http://54.84.168.9:8003/check/input` | `{"text": "..."}` |
| Guardrails Output | `http://54.84.168.9:8003/check/output` | `{"text": "..."}` |
| RAG | `http://54.84.168.9:8001/query` | `{"description": "..."}` |
| Image Analyser (URLs) | `http://54.84.168.9:8002/analyse/batch` | `{"image_urls": ["...", "..."]}` |
| Image Analyser (file) | `http://54.84.168.9:8002/analyse/upload` | multipart form (called from Streamlit, not n8n) |
| LangGraph Agent | `http://54.84.168.9:8004/agent/run` | `{"query": "..."}` |

These URLs are hard-coded in the workflow JSON. If the EC2 instance
moves, edit the URL field in each affected HTTP node and re-save the
workflow.

---

## Final report shape

The workflow returns this JSON to the Streamlit UI:

```json
{
  "property_type": "apartment",
  "routing_decision": "residential",
  "location": "Tel Aviv",
  "price_ils": 4200000,
  "num_rooms": 3,
  "size_sqm": 95,
  "key_features": ["central location", "recently renovated", "sea view"],
  "image_scores": [
    {
      "url": "https://example.com/photo1.jpg",
      "room_type": "living_room",
      "condition_score": 3,
      "confidence": 0.99
    }
  ],
  "similar_listings": [],
  "rag_insight": "...",
  "enrichment_notes": "",
  "confidence": 0.95,
  "markdown_summary": "..."
}
```

---

## Setup

### 1. Start n8n in Docker

```powershell
docker run -it --rm `
  --name n8n `
  -p 5678:5678 `
  -v n8n_data:/home/node/.n8n `
  docker.n8n.io/n8nio/n8n
```

Open <http://localhost:5678> and complete the n8n first-run setup.

### 2. Import the workflow

In n8n: top-right menu → **Import from File** → select
`AI_Property_Triage_v2.json`.

### 3. Configure the Gemini credential

The three Gemini Chat Model nodes share a single credential.

- Open any Gemini Chat Model node (e.g. `Gemini Chat Model — Extractor`).
- Click the credential field → **Create New**.
- Paste a Google AI Studio API key (free tier from
  <https://aistudio.google.com/app/apikey>).
- Save.

### 4. Activate the workflow

Toggle **Active** in the top-right of the workflow editor.

The webhook is now live at:

```
http://localhost:5678/webhook/property-triage
```

The Streamlit UI in Layer 1 posts to this URL.

---

## Implementation notes

### RAG tool body

The RAG tool uses a fixed body sourced directly from the webhook
description, not from the AI agent — this prevents the model from
truncating or rewriting the listing before retrieval:

```javascript
={{ JSON.stringify({
  description: $('Node 1 — Webhook Trigger').item.json.body.description || ''
}) }}
```

### Image URLs tool body

The `analyse_images` tool sanitises the URL list before posting to
`/analyse/batch`. It accepts both a JSON array and a newline / comma
separated string, trims whitespace, drops empties, and caps at 10:

```javascript
={{ JSON.stringify({
  image_urls: (() => {
    const raw = $('Node 1 — Webhook Trigger').item.json.body.image_urls;

    if (Array.isArray(raw)) {
      return raw
        .map(String)
        .map(s => s.trim())
        .filter(Boolean)
        .slice(0, 10);
    }

    return String(raw || '')
      .split(/[\n,]+/)
      .map(s => s.trim())
      .filter(Boolean)
      .slice(0, 10);
  })()
}) }}
```

### Uploaded image handling

Uploaded files are analysed in Streamlit (which calls
`/analyse/upload` directly per file) and shipped to n8n inside
`precomputed_image_scores`. The AI Agent copies those objects into the
final `image_scores` array unchanged — `analyse_images` is **not**
called when `precomputed_image_scores` is non-empty and `image_urls` is
empty.

### Image score normalisation (Node 5b)

The EC2 service can return either `image_url` or `url` as the key on
each score object. `Node 5b — Normalize Image Scores` guarantees the
final report always uses `url`:

```json
{
  "url": "...",
  "room_type": "...",
  "condition_score": 3,
  "confidence": 0.99
}
```

### LangGraph tool body

```javascript
={{ JSON.stringify({
  query: "Analyze this property listing: " +
         ($('Node 1 — Webhook Trigger').item.json.body.description || '')
}) }}
```

The agent decides on its own whether to invoke this tool; it is
optional and used only for deeper multi-step reasoning (e.g. renovation
planning).

---

## End-to-end test

With n8n active and the EC2 services up:

1. Open the Streamlit UI (Layer 1) → tab **Submit Listing**.
2. Paste an agent name and a property description (at least 10 chars).
3. Optionally paste one or more public image URLs **or** upload one or
   more `.jpg` / `.png` / `.webp` files.
4. Press **Submit**.

You should see the executions in n8n light up green node by node, and
Streamlit renders the structured report and saves it to Pinecone.

Typical end-to-end latency is a few seconds when the EC2 services are
warm; cold-start on the LLM-backed nodes can push it to a minute or two.

### Sample test listing

```
Beautiful 3-bedroom apartment in central Tel Aviv, 95 sqm, recently
renovated, sea view. Bright living room with large windows and access
to a balcony. Modern kitchen with premium appliances and two bathrooms.
Located near the beach, restaurants, and public transportation.
Asking price: 4,200,000 NIS.
```

Sample image URL:

```
https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg
```

Expected result:

- Routing: `residential`
- Location: Tel Aviv
- Rooms: 3, Size: 95 sqm, Price: ₪4,200,000
- Similar listings retrieved
- One image score with room_type + condition_score

---

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| Streamlit shows "n8n webhook not found (404)" | Workflow not active | Toggle **Active** in n8n |
| "Could not reach n8n" | Docker container stopped | Re-run the docker command above |
| `429 quota exceeded` on a Gemini node | Free-tier daily quota hit (20 / day per project) | Wait 24 hours or paste a new Gemini API key from a different Gmail |
| `Unable to connect to remote server` for an EC2 URL | The EC2 service is down | Ask the Layer 3 teammate to restart the Docker container |
| RAG returns 422 | Body schema mismatch | Confirm the body matches the JSON shapes in the EC2 endpoints table |
| `Could not replace placeholders in body` | Expression syntax error in JSON body | Re-paste the body from the Implementation notes section |

---

## Why each node is in the workflow

| # | Node | Why |
|---|---|---|
| 1 | Webhook Trigger | Receives the form POST from the Streamlit UI |
| 2 | Guardrails Input Check | Reject PII / abusive / off-topic submissions before any LLM cost |
| 3 | Input Pass/Reject | Routes based on the guardrails verdict |
| 3b | Reject Response | Returns 422 to the UI when input is rejected |
| 4 | Information Extractor | LangChain LLMChain that pulls structured fields from the free-text description |
| 4b | Parse Extracted JSON | Cleans and validates the LLM's JSON output |
| 5 | AI Agent | LangChain Agent that decides which tools to call (RAG / Image / LangGraph) |
| 5b | Normalize Image Scores | Reshapes batch image-analysis output into a consistent shape |
| 6 | Final Report LLM Chain | LangChain LLMChain that produces the user-facing report |
| 6b | Parse Report JSON | Validates the final report shape |
| 7 | Guardrails Output Check | Catches unsafe / low-confidence outputs |
| 7b | Output Pass/Flag | Sends flagged outputs to human review |
| 7c/d | Human Review path | Notifies reviewers and responds to the client |
| 8 | Property Type Router | Switch node deciding residential vs commercial |
| 8a/b/c | Team notifications + responses | Final hand-off and 200 response to the UI |

---

## Demo script (for the live presentation)

1. Start n8n in Docker.
2. Activate the `AI_Property_Triage_v2` workflow.
3. Start the Streamlit app.
4. Open the **Submit Listing** tab.
5. Enter the agent name.
6. Paste the sample property description (above).
7. Either paste the sample image URL **or** upload a local image file.
8. Press **Submit** and switch to the n8n tab to show the live execution
   light up node by node.
9. Switch back to Streamlit and show the final report.
10. Switch to the **AI Assistant** tab and run "show me my listings" —
    the new listing is now retrievable from the Pinecone vector store.

---

## Quality gates the workflow enforces

- **Input guardrails** — rejects PII, abusive text, and off-topic
  content before any LLM cost is incurred.
- **Schema validation** — every LLM output is parsed and validated in a
  code node; malformed JSON is caught here.
- **Output guardrails** — a second pass after report generation flags
  hallucinated or low-confidence reports for human review.
- **Routing accountability** — the residential vs commercial decision is
  captured in the report and visible to the agent who submitted the
  listing.

---

## Current status

| Component | Status |
|---|---|
| URL-based image analysis | Working |
| Uploaded image analysis | Working (via `precomputed_image_scores`) |
| RAG similar listings | Working |
| Final report generation | Working |
| Output guardrails | Working |
| Residential / commercial routing | Working |
| LangGraph agent | Optional, configured |
| Team notification endpoints | Mock placeholders (`127.0.0.1:9091` / `127.0.0.1:9092`) — replace with Slack / email / CRM in production |

---

## Future improvements

- Replace the mock notification endpoints with real Slack, email, CRM,
  or dashboard integrations.
- Add retry logic on transient EC2 failures.
- Add an admin dashboard for the human-review queue.
- Move the hard-coded EC2 IPs into n8n environment variables so the
  workflow JSON does not need editing when the IP changes.
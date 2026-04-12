# Analysis-Gen — Project Instructions

> **Purpose**: This document provides full context for an LLM agent working on this codebase. It covers architecture, tech stack, coding standards, API surface, data flow, and operational details.

---

## 1. Project Overview

**Analysis-Gen** is a Python **FastAPI** service that serves as the **AI analysis engine** for a sports data management platform. It receives CSV file URLs and natural language prompts, then returns text-based data analysis and generated chart images.

### Role in the System

```
┌──────────────┐       ┌──────────────┐       ┌──────────────────────┐
│   Frontend   │◄─────►│   API-KPI    │◄─────►│  Analysis Generator  │  ← THIS REPO
│  (Next.js)   │       │  (Go :8080)  │       │   (Python :9090)     │
│  :3000       │       │              │       │                      │
└──────────────┘       └──────┬───────┘       └──────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              ┌─────▼──┐ ┌───▼────┐ ┌──▼──────┐
              │MongoDB │ │ Pinata │ │SendGrid │
              │ :27017 │ │ (IPFS) │ │ (Email) │
              └────────┘ └────────┘ └─────────┘
```

- **Called by**: `API-kpi` (Go backend at `localhost:8080`), specifically by `cmd/api/handlers/analysis-gen.go` and `cmd/api/handlers/upload-ipfs.go`.
- **Does NOT**: Access MongoDB, authenticate users, or serve the frontend directly.
- **Runs on**: `http://127.0.0.1:9090`

---

## 2. Tech Stack

| Layer              | Technology                    | Notes                              |
| ------------------ | ----------------------------- | ---------------------------------- |
| **Language**       | Python 3                      |                                    |
| **Framework**      | FastAPI                       | ASGI via Uvicorn                   |
| **Server**         | Uvicorn                       | `--host 127.0.0.1 --port 9090`    |
| **Data Analysis**  | pandas, numpy                 | CSV parsing, statistics            |
| **Visualization**  | matplotlib, seaborn           | Chart generation (PNG → base64)    |
| **AI / LLM**       | OpenAI (`gpt-4o`)             | Primary analysis engine            |
| **AI Fallback**    | Hugging Face (DialoGPT)       | Secondary fallback                 |
| **HTTP Client**    | httpx                         | Async download of CSV files from IPFS |
| **Config**         | python-dotenv                 | `.env` file loading                |
| **File Handling**  | python-multipart              | For file upload endpoints          |
| **ML (available)** | scikit-learn                  | In requirements, not actively used |

---

## 3. Project Structure

```
analysis-gen/
├── src/
│   ├── main.py                # All application code (routes, logic, AI integration)
│   ├── player_files/          # Local storage for player-related files
│   ├── analysis.ipynb         # Jupyter notebook (exploratory)
│   └── analysis_cuiaba.ipynb  # Jupyter notebook (Cuiabá dataset analysis)
├── data/
│   ├── cuiaba.csv             # Sample dataset (~14MB)
│   ├── cuiaba.xlsx            # Same data in Excel format
│   └── soccer_players_season_stats.csv  # Small sample dataset
├── img/                       # Image assets
├── venv/                      # Python virtual environment
├── .env                       # Environment variables (API keys)
├── .gitignore
├── requirements.txt           # pip dependencies
├── upload-ipfs.go             # Go file (possibly legacy/copy from API-kpi)
└── README.md
```

> **Note**: All application logic is in a single file `src/main.py` (772 lines). There are no separate route/model modules despite the README suggesting otherwise.

---

## 4. API Routes

| Route                | Method | Request Format                              | Response Format         | Description                             |
| -------------------- | ------ | ------------------------------------------- | ----------------------- | --------------------------------------- |
| `/analysis-gen`      | POST   | JSON: `{ fileAddresses: [url], prompt: str }` | JSON (see §4.1)        | Main analysis endpoint                  |
| `/data-health-check` | POST   | Multipart: `file` (CSV upload)              | Plain text report       | Data quality check on uploaded CSV      |

### 4.1 `/analysis-gen` — Main Analysis Endpoint

**Request body** (JSON):
```json
{
  "fileAddresses": ["https://...ipfs-gateway.../hash1", "https://...ipfs-gateway.../hash2"],
  "prompt": "Who is the top scorer?"
}
```
- `fileAddresses`: List of IPFS gateway URLs pointing to CSV files.
- `prompt`: Natural language question about the data (supports Portuguese, English, Spanish).

**Response** (JSON):
```json
{
  "text_response": "The top scorer is...",
  "chart_base64": "iVBORw0KGgo...",
  "data_summary": {
    "rows": 500,
    "columns": 12,
    "column_names": ["name", "goals", ...],
    "numeric_columns": ["goals", "assists"],
    "categorical_columns": ["name", "team"],
    "files_processed": 2,
    "visualization_generated": true
  }
}
```
- `chart_base64`: Base64-encoded PNG image, or `null` if no chart was generated.

### 4.2 `/data-health-check` — Data Quality Check

**Request**: Multipart file upload (CSV).

**Response**: Plain text report including:
- Row/column counts
- Missing values per column
- Duplicate row detection
- Column data types
- Numeric column statistics (min, max, mean, median, outliers via IQR)
- Overall health verdict: GOOD / FAIR / POOR

---

## 5. Core Logic Flow

### 5.1 Analysis Pipeline (`/analysis-gen`)

```
1. Receive file URLs + prompt
2. Download CSV files from IPFS (via httpx async)
3. Parse each CSV into a pandas DataFrame
4. Combine DataFrames (concat vertically or merge horizontally)
5. Detect if visualization is requested
6. Analyze data:
   a. Try OpenAI GPT-4o (primary)
   b. Try Hugging Face DialoGPT (fallback)
   c. Use pandas-based rule analysis (final fallback)
7. Generate chart if needed:
   a. Try AI-generated matplotlib code (GPT-4o generates Python code, exec'd)
   b. Fall back to rule-based chart generation
8. Return JSON response
```

### 5.2 Language Detection

The service detects the user's language (Portuguese, English, Spanish) and responds in kind. Detection is keyword-based via `detect_language()`.

### 5.3 Visualization Request Detection

`is_visualization_request()` checks for keywords like "chart", "graph", "gráfico", "plotar", etc. Some analytical queries (containing "top", "best", "compare", etc.) also auto-generate charts.

### 5.4 Chart Generation

Two-tier approach:
1. **AI-generated**: GPT-4o generates Python matplotlib/seaborn code, which is `exec()`'d in a sandboxed namespace. The generated code saves to `output_chart.png`, which is then read and base64-encoded.
2. **Rule-based fallback**: Manual chart generation based on detected chart type keywords (bar, line, pie, scatter, histogram, heatmap, box plot).

---

## 6. Pydantic Models

```python
class URLRequest(BaseModel):
    fileAddresses: List[HttpUrl]  # IPFS gateway URLs
    prompt: str                   # Natural language question
```

---

## 7. Environment Variables (`.env`)

| Variable             | Purpose                                |
| -------------------- | -------------------------------------- |
| `OPENAI_API_KEY`     | OpenAI API key for GPT-4o analysis     |
| `HUGGINGFACE_API_KEY`| Hugging Face API key (fallback model)  |

---

## 8. Key Functions Reference

| Function                               | Purpose                                              |
| -------------------------------------- | ---------------------------------------------------- |
| `perform_data_health_check(df)`        | Generate plain-text data quality report              |
| `detect_language(text)`                | Detect language (pt/en) from keywords                |
| `is_visualization_request(prompt)`     | Check if prompt asks for a chart/graph               |
| `analyze_dataframe_with_openai(df, prompt)` | Primary analysis via GPT-4o                     |
| `analyze_dataframe_with_huggingface(df, prompt)` | Fallback analysis via Hugging Face          |
| `analyze_dataframe_fallback(df, prompt)` | Rule-based pandas analysis (no API needed)         |
| `generate_visualization_code_with_ai(df, prompt)` | GPT-4o generates matplotlib code           |
| `execute_visualization_code(df, code)` | Safely exec() AI-generated chart code                |
| `generate_chart_from_prompt(df, prompt)` | Orchestrates chart generation (AI → fallback)      |
| `generate_chart_fallback(df, prompt)`  | Rule-based chart generation                          |

---

## 9. How to Run

### Prerequisites
- **Python** 3.10+
- **API keys** in `.env` (at least `OPENAI_API_KEY` for full functionality)

### Steps
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn src.main:app --host 127.0.0.1 --port 9090
# OR
python src/main.py
```

The server starts on `http://127.0.0.1:9090`.

---

## 10. Coding Standards & Patterns

### General Patterns
- **Single-file architecture**: All logic in `src/main.py`. No separate route or model modules.
- **Async routes**: FastAPI async endpoints with `httpx.AsyncClient` for CSV downloads.
- **Error handling**: `HTTPException` with appropriate status codes (400, 500). `print()` for server-side logging.
- **AI fallback chain**: OpenAI → Hugging Face → pandas-based analysis. This ensures the service degrades gracefully without API keys.
- **Code execution**: AI-generated visualization code is `exec()`'d with a restricted globals namespace containing only `df`, `pd`, `plt`, `sns`, `np`, `io`, `base64`.

### Naming Conventions
- **Functions**: `snake_case` (e.g., `perform_data_health_check`, `detect_language`)
- **Variables**: `snake_case`
- **Constants**: Inline (no separate constants file)

### Response Format
- `/analysis-gen`: JSON with `text_response`, `chart_base64`, `data_summary`
- `/data-health-check`: Plain text (`PlainTextResponse`)

---

## 11. Key Considerations for Development

- **No authentication**: This service has no auth layer. It trusts all incoming requests (security is handled by the Go API layer).
- **`exec()` usage**: AI-generated code is executed via `exec()`. The namespace is limited but still poses a security consideration.
- **Single-file codebase**: Consider refactoring into separate modules (routes, services, utils) as the codebase grows.
- **No tests**: There are no unit or integration tests currently.
- **Bilingual responses**: The service responds in the same language as the user's prompt (Portuguese or English).
- **Large file handling**: The `cuiaba.csv` dataset is ~14MB. The service loads entire DataFrames into memory.
- **`upload-ipfs.go`**: This file appears to be a copy from the `API-kpi` repo and is not used by the Python service.

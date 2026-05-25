
# GenAI Deep Research Assistant

A clean, recruiter-friendly backend project that shows how a research assistant can answer a user question by searching a small knowledge base, retrieving relevant evidence, and producing a structured answer with citations.

This project is built to be honest and easy to run in one day. It works locally without paid API keys using a deterministic demo generator. It is also prepared for optional LLM integration later.

## Why this project matters

Many teams need AI tools that do more than produce a quick answer. A useful research assistant should:

- break a question into smaller research steps
- retrieve relevant information from trusted documents
- generate a structured answer
- show the sources used
- expose the workflow through a backend API

This project demonstrates that workflow with FastAPI, local retrieval, and a simple UI.

## Features

- FastAPI backend
- Local document retrieval using TF-IDF
- Multi-step research workflow
- Source citations in every answer
- Streamlit demo UI
- Docker support
- Pytest tests
- GitHub Actions CI workflow
- Clean structure for future LLM, RAG, and cloud deployment

## Project structure

```text
genai-deep-research-assistant/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── retriever.py
│   ├── generator.py
│   └── research.py
├── data/
│   └── knowledge_base/
├── frontend/
│   └── streamlit_app.py
├── tests/
│   └── test_api.py
├── docs/
│   └── architecture.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Quick start

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### 4. Test the research endpoint

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/research" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"How can a company build a reliable enterprise AI research assistant?\",\"top_k\":4}"
```

### 5. Run the Streamlit UI

Open a second terminal and run:

```bash
streamlit run frontend/streamlit_app.py
```

## Docker

```bash
docker build -t genai-deep-research-assistant .
docker run -p 8000:8000 genai-deep-research-assistant
```

Or:

```bash
docker compose up --build
```

## Example API response

```json
{
  "question": "How can a company build a reliable enterprise AI research assistant?",
  "answer": "A reliable enterprise AI research assistant should combine retrieval, controlled generation, source tracking, and deployment monitoring...",
  "citations": [
    {
      "source": "enterprise_ai_notes.md",
      "chunk_id": "enterprise_ai_notes.md::chunk_1",
      "score": 0.42
    }
  ]
}
```

## Resume version

Use this version in the resume after pushing the project to GitHub:

**Deep Research Assistant using LLM-ready Retrieval Workflow**  
Built a FastAPI-based research assistant that retrieves relevant document evidence, creates structured responses, and returns source citations. Designed the project with a modular backend, local retrieval, Docker support, tests, and a Streamlit demo UI to show how enterprise research workflows can be made transparent and reproducible.

## Future improvements

- Add Azure OpenAI or OpenAI based answer generation
- Add vector database support such as FAISS or Chroma
- Add authentication
- Add document upload support
- Add cloud deployment on AWS ECS or Azure App Service
- Add evaluation metrics for retrieval quality

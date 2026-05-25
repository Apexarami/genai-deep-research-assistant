# GenAI PDF Research Assistant
<img width="1878" height="849" alt="image" src="https://github.com/user-attachments/assets/8e42311d-c874-4bd4-9bad-119ea3b19e39" />

A local PDF research assistant built with FastAPI and Streamlit. Users can upload a PDF, ask questions about the document, retrieve relevant sections, and generate citation based answers.

The project works without paid LLM APIs. It also supports optional local AI generation through Ollama.

## What this project does

- Upload a PDF document
- Extract readable text page by page
- Split the document into searchable chunks
- Ask questions about the uploaded document
- Retrieve the most relevant sections
- Generate an answer grounded in document evidence
- Show page level citations
- Support simple explanation mode
- Optionally use a local Ollama model for smoother answers
- Run with FastAPI, Streamlit, Docker, and tests

## Why this project is useful

Many students, engineers, and researchers work with long PDF documents such as textbooks, reports, research papers, manuals, and lecture notes. This assistant helps users ask questions about those documents and receive answers based on the uploaded content.

Unlike a normal chatbot, this project does not pretend to know everything. It searches the uploaded document and uses only the retrieved evidence. This makes the answer more transparent and easier to verify.

## Architecture

```text
User uploads PDF
    |
    v
PDF text extraction
    |
    v
Page based chunking
    |
    v
TF IDF retrieval
    |
    v
Relevant evidence chunks
    |
    +--> Extractive answer mode
    |
    +--> Optional Ollama local LLM answer mode
    |
    v
Final answer with page citations
```

## Quick start

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Start the backend

```powershell
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### 4. Start the frontend

Open a second PowerShell window:

```powershell
cd C:\Users\apexa\desktop\genai-deep-research-assistant
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Optional local AI with Ollama

Install Ollama on Windows, then download a small local model:

```powershell
ollama pull llama3.2:3b
```

Test it:

```powershell
ollama run llama3.2:3b
```

If Ollama is running, the app can use it for natural answers. If not, the app still works in extractive answer mode.

## Example questions

After uploading a semiconductor textbook or lecture PDF, ask:

```text
What is semiconductor doping?
```

```text
Explain the difference between n type and p type semiconductors.
```

```text
Explain this topic in a more understandable way.
```

## API endpoints

```text
GET  /health
GET  /ollama/status
POST /pdf/upload
POST /pdf/ask
```

## Limitations

- Scanned image only PDFs may not work because OCR is not included.
- The answer quality depends on the quality of text extracted from the PDF.
- Ollama generation depends on the local machine resources and downloaded model.
- The current document index is stored in memory and resets when the backend restarts.

## Future improvements

- Add persistent document storage
- Add OCR for scanned PDFs
- Add FAISS or Chroma vector search
- Add user authentication
- Add chat history
- Add cloud deployment

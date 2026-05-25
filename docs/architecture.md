# Architecture

The project follows a simple service structure.

```text
User / UI
   |
   v
Streamlit demo or API client
   |
   v
FastAPI backend
   |
   +--> Research workflow
          |
          +--> Step planning
          |
          +--> Local document retrieval
          |
          +--> Answer generation
          |
          +--> Citation response
```

## Main components

### FastAPI backend

The backend exposes the project as an API. This makes the logic usable from a web app, chatbot, dashboard, or another backend service.

### Local retriever

The retriever loads text and markdown files from the knowledge base folder. It splits the files into small chunks and ranks them using TF-IDF cosine similarity.

This is simple but useful for a local MVP. A production version could replace this with FAISS, Chroma, Azure AI Search, or another vector database.

### Research workflow

The workflow creates three research steps from the user question. Each step is searched separately. The best evidence chunks are merged, ranked, and sent to the answer generator.

### Answer generator

The current generator runs without external API keys. It creates a structured answer from the retrieved evidence. This makes the project easy to test and share.

A production version can replace this class with Azure OpenAI, OpenAI, or another LLM provider.

## Why this design is useful

The project separates retrieval, generation, and API logic. This makes it easier to test, improve, and deploy. It also makes the project believable as an enterprise AI prototype because every answer includes source citations.

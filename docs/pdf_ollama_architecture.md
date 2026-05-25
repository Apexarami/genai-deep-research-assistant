# PDF and Ollama Architecture

This version upgrades the project from a fixed local knowledge base demo to an upload based document research assistant.

## Flow

1. The user uploads a PDF from the Streamlit interface.
2. The FastAPI backend extracts text from the PDF page by page.
3. The document is split into chunks while keeping page numbers.
4. A TF IDF retriever ranks chunks against the user question.
5. The strongest evidence chunks are selected.
6. If Ollama is available, a local LLM creates a natural answer from the evidence.
7. If Ollama is not available, the app creates an extractive answer from the evidence.
8. The frontend displays the final answer and page based citations.

## Why Ollama

Ollama allows the project to run local language models without paid API keys. This is useful for privacy, cost control, and local experimentation.

## Why fallback mode matters

The app should still work even if a local model is not installed. For this reason, extractive mode is kept as a fallback. This makes the project easier to run on different laptops and easier for recruiters to test.

from functools import lru_cache
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.document_qa import DocumentAnswerService, DocumentQAStore
from app.generator import LocalAnswerGenerator
from app.ollama_client import OllamaClient
from app.pdf_reader import extract_pdf_pages
from app.research import ResearchWorkflow
from app.retriever import LocalDocumentRetriever
from app.schemas import (
    Citation, PdfAskRequest, PdfAskResponse, PdfDeepResearchRequest,
    PdfDeepResearchResponse, PdfEvidence, PdfUploadResponse,
    ResearchRequest, ResearchResponse, SearchRequest, SearchResult,
)

app = FastAPI(
    title="GenAI PDF Deep Research Assistant",
    description="PDF Q&A and deep document research with local retrieval, citations, and optional Ollama.",
    version="3.0.0",
)

document_store = DocumentQAStore()
ollama_client = OllamaClient()
document_answer_service = DocumentAnswerService(document_store, ollama_client)

@lru_cache
def get_retriever() -> LocalDocumentRetriever:
    return LocalDocumentRetriever(settings.knowledge_base_dir)

@lru_cache
def get_workflow() -> ResearchWorkflow:
    return ResearchWorkflow(get_retriever(), LocalAnswerGenerator())

def to_citation(item) -> Citation:
    preview = item.text[:220].replace("\n", " ") + ("..." if len(item.text) > 220 else "")
    return Citation(source=item.source, chunk_id=item.chunk_id, score=item.score, text_preview=preview)

def to_pdf_evidence(item) -> PdfEvidence:
    preview = item.text[:500].replace("\n", " ") + ("..." if len(item.text) > 500 else "")
    return PdfEvidence(source=item.document_name, page_number=item.page_number, chunk_id=item.chunk_id, score=item.score, text_preview=preview)

@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "app": "GenAI PDF Deep Research Assistant", "ollama_available": ollama_client.is_available()}

@app.get("/ollama/status")
def ollama_status() -> dict:
    return {"available": ollama_client.is_available(), "base_url": ollama_client.base_url, "suggested_model": "llama3.2:3b"}

@app.get("/documents")
def list_uploaded_documents() -> dict:
    return {"documents": document_store.list_documents()}

@app.post("/pdf/upload", response_model=PdfUploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> PdfUploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    try:
        pages = extract_pdf_pages(await file.read())
        index = document_store.add_document(file.filename, pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PdfUploadResponse(doc_id=index.doc_id, document_name=index.document_name, page_count=len(index.pages), chunk_count=len(index.chunks))

@app.post("/pdf/ask", response_model=PdfAskResponse)
def ask_pdf(request: PdfAskRequest) -> PdfAskResponse:
    try:
        mode, answer, evidence = document_answer_service.answer(request.doc_id, request.question, request.style, request.top_k, request.use_ollama, request.model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PdfAskResponse(doc_id=request.doc_id, question=request.question, answer_mode=mode, answer=answer, evidence=[to_pdf_evidence(e) for e in evidence])

@app.post("/pdf/deep-research", response_model=PdfDeepResearchResponse)
def deep_research_pdf(request: PdfDeepResearchRequest) -> PdfDeepResearchResponse:
    try:
        result = document_answer_service.deep_research(request.doc_id, request.question, request.style, request.top_k, request.use_ollama, request.model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PdfDeepResearchResponse(
        doc_id=request.doc_id,
        question=request.question,
        answer_mode=result.answer_mode,
        confidence=result.confidence,
        research_plan=result.research_plan,
        search_queries=result.search_queries,
        answer=result.final_answer,
        evidence=[to_pdf_evidence(e) for e in result.evidence],
    )

# Backward-compatible demo endpoints
@app.get("/kb/documents")
def list_knowledge_base_documents() -> dict:
    r = get_retriever()
    sources = sorted({c.source for c in r.chunks})
    return {"document_count": len(sources), "chunk_count": len(r.chunks), "sources": sources}

@app.post("/search", response_model=SearchResult)
def search_documents(request: SearchRequest) -> SearchResult:
    results = get_retriever().search(request.query, top_k=request.top_k)
    return SearchResult(query=request.query, results=[to_citation(i) for i in results])

@app.post("/research", response_model=ResearchResponse)
def research_question(request: ResearchRequest) -> ResearchResponse:
    steps, answer, citations = get_workflow().run(request.question, top_k=request.top_k)
    return ResearchResponse(question=request.question, research_steps=steps, answer=answer, citations=[to_citation(i) for i in citations])

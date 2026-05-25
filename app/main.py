from functools import lru_cache

from fastapi import FastAPI

from app.config import settings
from app.generator import LocalAnswerGenerator
from app.research import ResearchWorkflow
from app.retriever import LocalDocumentRetriever
from app.schemas import (
    Citation,
    ResearchRequest,
    ResearchResponse,
    SearchRequest,
    SearchResult,
)


app = FastAPI(
    title=settings.app_name,
    description="FastAPI backend for an LLM-ready deep research assistant with retrieval and citations.",
    version="1.0.0",
)


@lru_cache
def get_retriever() -> LocalDocumentRetriever:
    return LocalDocumentRetriever(settings.knowledge_base_dir)


@lru_cache
def get_workflow() -> ResearchWorkflow:
    return ResearchWorkflow(
        retriever=get_retriever(),
        generator=LocalAnswerGenerator(),
    )


def to_citation(item) -> Citation:
    preview = item.text[:220].replace("\n", " ")
    if len(item.text) > 220:
        preview += "..."

    return Citation(
        source=item.source,
        chunk_id=item.chunk_id,
        score=item.score,
        text_preview=preview,
    )


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm_provider": settings.llm_provider,
    }


@app.get("/documents")
def list_documents() -> dict:
    retriever = get_retriever()
    sources = sorted({chunk.source for chunk in retriever.chunks})
    return {
        "document_count": len(sources),
        "chunk_count": len(retriever.chunks),
        "sources": sources,
    }


@app.post("/search", response_model=SearchResult)
def search_documents(request: SearchRequest) -> SearchResult:
    retriever = get_retriever()
    results = retriever.search(request.query, top_k=request.top_k)

    return SearchResult(
        query=request.query,
        results=[to_citation(item) for item in results],
    )


@app.post("/research", response_model=ResearchResponse)
def research_question(request: ResearchRequest) -> ResearchResponse:
    workflow = get_workflow()
    steps, answer, citations = workflow.run(request.question, top_k=request.top_k)

    return ResearchResponse(
        question=request.question,
        research_steps=steps,
        answer=answer,
        citations=[to_citation(item) for item in citations],
    )

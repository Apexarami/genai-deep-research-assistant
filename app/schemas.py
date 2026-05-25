from pydantic import BaseModel, Field
from typing import List

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=10)

class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float
    text_preview: str

class SearchResult(BaseModel):
    query: str
    results: List[Citation]

class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=5)
    top_k: int = Field(4, ge=1, le=10)

class ResearchResponse(BaseModel):
    question: str
    research_steps: List[str]
    answer: str
    citations: List[Citation]

class PdfUploadResponse(BaseModel):
    doc_id: str
    document_name: str
    page_count: int
    chunk_count: int

class PdfAskRequest(BaseModel):
    doc_id: str
    question: str = Field(..., min_length=3)
    style: str = "technical"
    top_k: int = Field(6, ge=1, le=20)
    use_ollama: bool = True
    model: str = "llama3.2:3b"

class PdfDeepResearchRequest(BaseModel):
    doc_id: str
    question: str = Field(..., min_length=3)
    style: str = "technical"
    top_k: int = Field(10, ge=4, le=20)
    use_ollama: bool = True
    model: str = "llama3.2:3b"

class PdfEvidence(BaseModel):
    source: str
    page_number: int
    chunk_id: str
    score: float
    text_preview: str

class PdfAskResponse(BaseModel):
    doc_id: str
    question: str
    answer_mode: str
    answer: str
    evidence: List[PdfEvidence]

class PdfDeepResearchResponse(BaseModel):
    doc_id: str
    question: str
    answer_mode: str
    confidence: str
    research_plan: List[str]
    search_queries: List[str]
    answer: str
    evidence: List[PdfEvidence]

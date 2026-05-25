from pydantic import BaseModel, Field
from typing import List


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User search query")
    top_k: int = Field(4, ge=1, le=10, description="Number of chunks to return")


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float
    text_preview: str


class SearchResult(BaseModel):
    query: str
    results: List[Citation]


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Research question")
    top_k: int = Field(4, ge=1, le=10, description="Number of evidence chunks to retrieve")


class ResearchResponse(BaseModel):
    question: str
    research_steps: List[str]
    answer: str
    citations: List[Citation]

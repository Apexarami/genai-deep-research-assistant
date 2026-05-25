from dataclasses import dataclass
from pathlib import Path
from typing import List
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DocumentChunk:
    source: str
    chunk_id: str
    text: str


@dataclass
class RetrievedChunk:
    source: str
    chunk_id: str
    text: str
    score: float


class LocalDocumentRetriever:
    """Small local retriever for demo-grade RAG workflows.

    It loads markdown and text files, splits them into chunks, and ranks chunks
    with TF-IDF cosine similarity. This is intentionally simple so the project
    can run locally without a paid vector database.
    """

    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = None
        self._load_documents()

    def _load_documents(self) -> None:
        if not self.knowledge_base_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {self.knowledge_base_dir}")

        files = sorted(
            list(self.knowledge_base_dir.glob("*.md")) +
            list(self.knowledge_base_dir.glob("*.txt"))
        )

        if not files:
            raise ValueError(f"No .md or .txt files found in {self.knowledge_base_dir}")

        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            paragraphs = self._split_into_chunks(text)
            for index, chunk_text in enumerate(paragraphs, start=1):
                chunk_id = f"{file_path.name}::chunk_{index}"
                self.chunks.append(
                    DocumentChunk(
                        source=file_path.name,
                        chunk_id=chunk_id,
                        text=chunk_text,
                    )
                )

        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    @staticmethod
    def _split_into_chunks(text: str, max_words: int = 120) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)

        chunks = []
        current = []

        for sentence in sentences:
            current.append(sentence)
            if len(" ".join(current).split()) >= max_words:
                chunks.append(" ".join(current).strip())
                current = []

        if current:
            chunks.append(" ".join(current).strip())

        return [chunk for chunk in chunks if len(chunk.split()) >= 8]

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self.matrix is None or not self.chunks:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=round(score, 4),
                )
            )

        return results

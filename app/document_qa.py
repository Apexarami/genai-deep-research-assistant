from dataclasses import dataclass
from typing import Dict, List, Tuple
from uuid import uuid4
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ollama_client import OllamaClient
from app.pdf_reader import PdfPage


@dataclass
class DocumentChunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str


@dataclass
class EvidenceChunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str
    score: float


@dataclass
class DeepResearchResult:
    answer_mode: str
    confidence: str
    research_plan: List[str]
    search_queries: List[str]
    final_answer: str
    evidence: List[EvidenceChunk]


class DocumentIndex:
    def __init__(self, doc_id: str, document_name: str, pages: List[PdfPage]):
        self.doc_id = doc_id
        self.document_name = document_name
        self.pages = pages
        self.chunks = self._chunk_pages(pages)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def _chunk_pages(self, pages: List[PdfPage], max_words: int = 190, overlap: int = 45) -> List[DocumentChunk]:
        chunks = []
        for page in pages:
            words = page.text.split()
            start = 0
            local = 1
            while start < len(words):
                end = min(start + max_words, len(words))
                text = " ".join(words[start:end]).strip()
                if len(text.split()) >= 18:
                    chunks.append(DocumentChunk(
                        chunk_id=f"{self.document_name}::p{page.page_number}::c{local}",
                        document_name=self.document_name,
                        page_number=page.page_number,
                        text=text,
                    ))
                if end == len(words):
                    break
                start = max(end - overlap, start + 1)
                local += 1
        return chunks

    def search(self, query: str, top_k: int = 8) -> List[EvidenceChunk]:
        if not self.chunks:
            return []
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix).flatten()
        order = scores.argsort()[::-1][:top_k]
        out = []
        for idx in order:
            c = self.chunks[idx]
            out.append(EvidenceChunk(c.chunk_id, c.document_name, c.page_number, c.text, round(float(scores[idx]), 4)))
        return out

    def overview_chunks(self, max_chunks: int = 8) -> List[EvidenceChunk]:
        selected = self.chunks[:max_chunks]
        return [EvidenceChunk(c.chunk_id, c.document_name, c.page_number, c.text, 1.0) for c in selected]


class DocumentQAStore:
    def __init__(self):
        self._documents: Dict[str, DocumentIndex] = {}

    def add_document(self, document_name: str, pages: List[PdfPage]) -> DocumentIndex:
        doc_id = str(uuid4())
        index = DocumentIndex(doc_id, document_name, pages)
        self._documents[doc_id] = index
        return index

    def get_document(self, doc_id: str) -> DocumentIndex:
        if doc_id not in self._documents:
            raise KeyError(f"Unknown document id: {doc_id}")
        return self._documents[doc_id]

    def list_documents(self) -> List[dict]:
        return [{
            "doc_id": doc_id,
            "document_name": idx.document_name,
            "page_count": len(idx.pages),
            "chunk_count": len(idx.chunks),
        } for doc_id, idx in self._documents.items()]


class DocumentAnswerService:
    def __init__(self, store: DocumentQAStore, ollama_client: OllamaClient):
        self.store = store
        self.ollama = ollama_client
        self.minimum_relevance_score = 0.01

    def _is_overview_question(self, question: str) -> bool:
        q = question.lower().strip()
        return any(p in q for p in [
            "what is this document about", "what is this pdf about", "summarize this document",
            "summary of this document", "explain this document", "main idea", "main topic",
            "overview", "what does this document say", "what does this pdf say"
        ])

    def _clean_query(self, question: str) -> str:
        q = re.sub(r"[^a-zA-Z0-9\s]", " ", question.lower())
        stop = {"what", "is", "are", "the", "this", "that", "about", "explain", "tell", "me", "please", "document", "pdf", "in", "a", "an", "of", "to", "and", "from", "how", "why"}
        return " ".join(w for w in q.split() if w not in stop and len(w) > 2)

    def _query_variants(self, question: str) -> List[str]:
        core = self._clean_query(question)
        queries = [question]
        if core:
            queries += [core, f"definition explanation {core}", f"key points examples process {core}", f"advantages limitations {core}"]
        unique = []
        for q in queries:
            if q and q not in unique:
                unique.append(q)
        return unique

    def _retrieve(self, index: DocumentIndex, question: str, top_k: int, deep: bool) -> Tuple[List[str], List[EvidenceChunk]]:
        if self._is_overview_question(question):
            return ["document overview using first important chunks"], index.overview_chunks(max_chunks=top_k)
        queries = self._query_variants(question)
        collected: Dict[str, EvidenceChunk] = {}
        per_query = top_k if deep else max(4, top_k // max(1, len(queries)))
        for q in queries:
            for item in index.search(q, top_k=per_query):
                if item.score < self.minimum_relevance_score:
                    continue
                if item.chunk_id not in collected or item.score > collected[item.chunk_id].score:
                    collected[item.chunk_id] = item
        ranked = sorted(collected.values(), key=lambda x: x.score, reverse=True)
        if not ranked:
            ranked = index.overview_chunks(max_chunks=min(top_k, 5))
        return queries, ranked[:top_k]

    def answer(self, doc_id: str, question: str, style: str = "technical", top_k: int = 6, use_ollama: bool = True, model: str = "llama3.2:3b") -> Tuple[str, str, List[EvidenceChunk]]:
        idx = self.store.get_document(doc_id)
        _, evidence = self._retrieve(idx, question, top_k, deep=False)
        if use_ollama and self.ollama.is_available():
            return "ollama", self._ollama_answer(question, evidence, style, model, deep=False), evidence
        return "extractive", self._extractive_answer(question, evidence, style), evidence

    def deep_research(self, doc_id: str, question: str, style: str = "technical", top_k: int = 10, use_ollama: bool = True, model: str = "llama3.2:3b") -> DeepResearchResult:
        idx = self.store.get_document(doc_id)
        plan = self._research_plan(question)
        queries, evidence = self._retrieve(idx, question, top_k, deep=True)
        confidence = self._confidence(evidence)
        if use_ollama and self.ollama.is_available():
            answer = self._ollama_deep_answer(question, evidence, plan, style, model)
            mode = "ollama_deep_research"
        else:
            answer = self._extractive_deep_answer(question, evidence, plan, style)
            mode = "extractive_deep_research"
        return DeepResearchResult(mode, confidence, plan, queries, answer, evidence)

    def _research_plan(self, question: str) -> List[str]:
        if self._is_overview_question(question):
            return [
                "Identify the document type and main subject.",
                "Read the opening sections for context.",
                "Extract the main points and important terms.",
                "Create a clear overview with page references.",
            ]
        core = self._clean_query(question) or question
        return [
            f"Understand the question: {question}",
            f"Search for direct definitions and explanations related to: {core}",
            f"Search for supporting details, examples, causes, effects, or steps related to: {core}",
            "Remove repeated or weak sections.",
            "Write one final answer using only document evidence.",
        ]

    def _confidence(self, evidence: List[EvidenceChunk]) -> str:
        if not evidence:
            return "low"
        best = evidence[0].score
        pages = len({e.page_number for e in evidence})
        if best >= 0.20 and pages >= 2:
            return "high"
        if best >= 0.08 or pages >= 2:
            return "medium"
        return "low"

    def _style_instruction(self, style: str) -> str:
        if style == "simple":
            return "Write in simple beginner friendly language. Use short sentences and avoid heavy words."
        if style == "exam":
            return "Write like exam notes with definition, key points, explanation, and final takeaway."
        return "Write a clear technical answer that is structured and faithful to the document."

    def _evidence_text(self, evidence: List[EvidenceChunk], limit: int = 900) -> str:
        blocks = []
        for i, e in enumerate(evidence, 1):
            text = e.text[:limit]
            blocks.append(f"[Evidence {i} | Page {e.page_number} | Score {e.score}]\n{text}")
        return "\n\n".join(blocks)

    def _ollama_answer(self, question: str, evidence: List[EvidenceChunk], style: str, model: str, deep: bool) -> str:
        prompt = f"""
You are a document question answering assistant.
Rules:
- Answer only from the provided document evidence.
- Do not use outside knowledge.
- If evidence is weak, say what is missing.
- Mention page numbers when useful.
- Do not invent facts.
Style: {self._style_instruction(style)}
Question: {question}
Evidence:\n{self._evidence_text(evidence)}
Final answer:
"""
        return self.ollama.generate(model=model, prompt=prompt, temperature=0.2, num_predict=450)

    def _ollama_deep_answer(self, question: str, evidence: List[EvidenceChunk], plan: List[str], style: str, model: str) -> str:
        plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1))
        prompt = f"""
You are a deep document research assistant.
Use only the uploaded document evidence. Do not use outside knowledge.
Follow the research plan and create one complete answer.
Style: {self._style_instruction(style)}
Question: {question}
Research plan:\n{plan_text}
Evidence:\n{self._evidence_text(evidence, limit=1100)}
Write the answer with:
1. Direct answer
2. Explanation
3. Key points
4. Page references used
5. What the document does not clearly explain, if anything
"""
        return self.ollama.generate(model=model, prompt=prompt, temperature=0.15, num_predict=500)

    def _extractive_answer(self, question: str, evidence: List[EvidenceChunk], style: str) -> str:
        if not evidence:
            return "I could not find relevant evidence in the uploaded document."
        lines = [f"Question: {question}", "Most relevant evidence from the document:"]
        for i, e in enumerate(evidence, 1):
            lines.append(f"{i}. Page {e.page_number}: {e.text[:650]}...")
        lines.append("Note: This is extractive mode. Turn on Ollama for smoother generated answers.")
        return "\n\n".join(lines)

    def _extractive_deep_answer(self, question: str, evidence: List[EvidenceChunk], plan: List[str], style: str) -> str:
        plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1))
        ev_text = "\n".join(f"{i}. Page {e.page_number}, score {e.score}: {e.text[:750]}..." for i, e in enumerate(evidence, 1))
        pages = ", ".join(str(p) for p in sorted({e.page_number for e in evidence}))
        return f"Deep research question: {question}\n\nResearch plan followed:\n{plan_text}\n\nMost relevant evidence:\n{ev_text}\n\nPage references used: {pages}\n\nNote: This is extractive deep research mode. With Ollama running, the app generates a smoother final explanation."

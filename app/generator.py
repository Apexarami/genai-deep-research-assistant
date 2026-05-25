from typing import List

from app.retriever import RetrievedChunk


class LocalAnswerGenerator:
    """Deterministic answer generator for local demo mode.

    This class does not pretend to be a real LLM. It creates a structured answer
    from the retrieved evidence so the project can be run without API keys. The
    interface is intentionally simple and can later be replaced by Azure OpenAI,
    OpenAI, or another model provider.
    """

    def generate(self, question: str, evidence: List[RetrievedChunk]) -> str:
        if not evidence:
            return (
                "I could not find enough evidence in the local knowledge base to answer this question. "
                "Add more documents to data/knowledge_base or use a more specific question."
            )

        strongest_sources = ", ".join(sorted({item.source for item in evidence}))

        evidence_points = []
        for item in evidence[:4]:
            clean_text = item.text.strip()
            if len(clean_text) > 280:
                clean_text = clean_text[:280].rsplit(" ", 1)[0] + "..."
            evidence_points.append(f"- From {item.source}: {clean_text}")

        answer = [
            f"Research question: {question}",
            "",
            "Structured answer:",
            (
                "A reliable AI research assistant should combine retrieval, controlled answer generation, "
                "source tracking, and a clear service interface. The retrieval layer helps the system ground "
                "answers in available documents instead of relying only on model memory. The API layer makes "
                "the workflow reusable for a dashboard, chatbot, or enterprise application."
            ),
            "",
            "Evidence used:",
            *evidence_points,
            "",
            (
                f"Overall, the strongest evidence came from: {strongest_sources}. "
                "For a production version, the next step would be adding an LLM provider, user authentication, "
                "monitoring, and evaluation of answer quality."
            ),
        ]

        return "\n".join(answer)

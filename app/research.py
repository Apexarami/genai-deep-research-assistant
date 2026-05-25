from typing import List

from app.generator import LocalAnswerGenerator
from app.retriever import LocalDocumentRetriever, RetrievedChunk


class ResearchWorkflow:
    """Multi-step research workflow.

    The workflow is intentionally simple:
    1. Create small research steps from the user question.
    2. Retrieve evidence for each step.
    3. Merge and rank evidence.
    4. Generate a structured answer with citations.
    """

    def __init__(self, retriever: LocalDocumentRetriever, generator: LocalAnswerGenerator):
        self.retriever = retriever
        self.generator = generator

    def create_research_steps(self, question: str) -> List[str]:
        base_question = question.strip().rstrip("?")

        return [
            f"Define the main technical problem in: {base_question}",
            f"Find implementation components related to: {base_question}",
            f"Identify reliability and deployment concerns for: {base_question}",
        ]

    def run(self, question: str, top_k: int = 4) -> tuple[List[str], str, List[RetrievedChunk]]:
        steps = self.create_research_steps(question)
        collected: List[RetrievedChunk] = []

        for step in steps:
            collected.extend(self.retriever.search(step, top_k=top_k))

        deduplicated = {}
        for item in collected:
            if item.chunk_id not in deduplicated or item.score > deduplicated[item.chunk_id].score:
                deduplicated[item.chunk_id] = item

        ranked = sorted(deduplicated.values(), key=lambda item: item.score, reverse=True)[:top_k]
        answer = self.generator.generate(question, ranked)

        return steps, answer, ranked

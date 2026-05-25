from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_documents_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] >= 1
    assert data["chunk_count"] >= 1


def test_research_endpoint():
    response = client.post(
        "/research",
        json={
            "question": "How can companies build reliable AI research assistants?",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) >= 1

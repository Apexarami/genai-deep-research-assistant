from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_uploaded_documents_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_knowledge_base_documents_endpoint():
    response = client.get("/kb/documents")
    assert response.status_code == 200
    data = response.json()
    assert "document_count" in data
    assert "chunk_count" in data
    assert "sources" in data
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
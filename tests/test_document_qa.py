from app.document_qa import DocumentQAStore
from app.pdf_reader import PdfPage


def test_document_store_search():
    store = DocumentQAStore()
    index = store.add_document(
        "sample.pdf",
        [
            PdfPage(
                page_number=1,
                text=(
                    "Semiconductor doping adds impurity atoms to change electrical behavior. "
                    "N type material has extra electrons and P type material has holes."
                ),
            )
        ],
    )

    results = index.search("What is semiconductor doping?", top_k=2)

    assert index.document_name == "sample.pdf"
    assert len(results) >= 1
    assert results[0].page_number == 1

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="GenAI PDF Deep Research Assistant", layout="wide")
st.title("GenAI PDF Deep Research Assistant")
st.write("Upload a PDF, ask questions, and run deep research over the document. The app uses document evidence, page citations, and optional local Ollama.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("Backend URL", value=API_URL)
    use_ollama = st.checkbox("Use local Ollama if available", value=True)
    model = st.text_input("Ollama model", value="llama3.2:3b")
    style = st.selectbox("Answer style", ["technical", "simple", "exam"], index=0)
    mode = st.radio("Mode", ["Quick Q&A", "Deep Research"], index=1)
    top_k = st.slider("Evidence chunks", 4, 20, 10)
    if st.button("Check Ollama status"):
        try:
            data = requests.get(f"{api_url}/ollama/status", timeout=10).json()
            st.success("Ollama is available.") if data.get("available") else st.warning("Ollama is not available. Extractive mode will be used.")
        except requests.RequestException as exc:
            st.error(f"Could not reach backend: {exc}")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Reading and indexing PDF..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post(f"{api_url}/pdf/upload", files=files, timeout=420)
                res.raise_for_status()
                data = res.json()
                st.session_state["doc_id"] = data["doc_id"]
                st.session_state["document_name"] = data["document_name"]
                st.session_state["page_count"] = data["page_count"]
                st.session_state["chunk_count"] = data["chunk_count"]
                st.success(f"Processed {data['document_name']} with {data['page_count']} pages and {data['chunk_count']} chunks.")
            except requests.RequestException as exc:
                st.error(f"PDF upload failed: {exc}")
                if getattr(exc, "response", None) is not None:
                    st.code(exc.response.text)

if "doc_id" in st.session_state:
    st.info(f"Active document: {st.session_state['document_name']} | {st.session_state.get('page_count')} pages | {st.session_state.get('chunk_count')} chunks")
    default_q = "What is this document about?" if mode == "Deep Research" else "Explain the main concept of this document."
    question = st.text_area("Ask a question about the uploaded document", value=default_q, height=120)
    col1, col2 = st.columns([1, 1])
    run_button = col1.button("Run deep research" if mode == "Deep Research" else "Run document Q&A", type="primary")
    simple_button = col2.button("Explain more simply")

    if run_button or simple_button:
        endpoint = "/pdf/deep-research" if mode == "Deep Research" else "/pdf/ask"
        selected_style = "simple" if simple_button else style
        with st.spinner("Running deep research..." if mode == "Deep Research" else "Searching document..."):
            try:
                res = requests.post(
                    f"{api_url}{endpoint}",
                    json={"doc_id": st.session_state["doc_id"], "question": question, "style": selected_style, "top_k": top_k, "use_ollama": use_ollama, "model": model},
                    timeout=420,
                )
                res.raise_for_status()
                data = res.json()
            except requests.RequestException as exc:
                st.error(f"Question answering failed: {exc}")
                if getattr(exc, "response", None) is not None:
                    st.code(exc.response.text)
                st.stop()

        st.subheader("Answer")
        st.caption(f"Answer mode: {data['answer_mode']}")
        if "confidence" in data:
            c = data["confidence"]
            if c == "high": st.success(f"Evidence confidence: {c}")
            elif c == "medium": st.warning(f"Evidence confidence: {c}")
            else: st.error(f"Evidence confidence: {c}")
        st.write(data["answer"])

        if mode == "Deep Research":
            with st.expander("Research plan"):
                for step in data.get("research_plan", []): st.write(f"- {step}")
            with st.expander("Search queries used"):
                for query in data.get("search_queries", []): st.write(f"- {query}")

        st.subheader("Evidence from document")
        if not data["evidence"]:
            st.warning("No strong evidence was found for this question.")
        else:
            for i, item in enumerate(data["evidence"], 1):
                with st.expander(f"Evidence {i}: Page {item['page_number']} | score {item['score']}"):
                    st.write(item["text_preview"])
                    st.caption(item["chunk_id"])
else:
    st.warning("Upload and process a PDF first.")

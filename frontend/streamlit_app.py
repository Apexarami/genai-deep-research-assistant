import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="GenAI Deep Research Assistant", layout="wide")

st.title("GenAI Deep Research Assistant")
st.write(
    "Ask a research question. The app retrieves evidence from a local knowledge base "
    "and returns a structured answer with citations."
)

question = st.text_area(
    "Research question",
    value="How can a company build a reliable enterprise AI research assistant?",
    height=100,
)

top_k = st.slider("Number of evidence chunks", min_value=1, max_value=10, value=4)

if st.button("Run research"):
    if not question.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Researching..."):
            response = requests.post(
                f"{API_URL}/research",
                json={"question": question, "top_k": top_k},
                timeout=30,
            )

        if response.status_code != 200:
            st.error(f"API error: {response.status_code}")
            st.code(response.text)
        else:
            data = response.json()

            st.subheader("Answer")
            st.text(data["answer"])

            st.subheader("Research steps")
            for step in data["research_steps"]:
                st.write(f"- {step}")

            st.subheader("Citations")
            for citation in data["citations"]:
                st.markdown(
                    f"**{citation['source']}** | `{citation['chunk_id']}` | score: `{citation['score']}`"
                )
                st.write(citation["text_preview"])

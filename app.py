"""
YouTube Video RAG — Streamlit App
Ask questions about any YouTube video using its transcript as context.
"""

import os
import re
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ---------- Page config ----------
st.set_page_config(page_title="YouTube RAG Q&A", page_icon="🎥", layout="centered")
st.title("🎥 YouTube Video Q&A (RAG)")
st.caption("Paste a YouTube link, then ask questions — answered only from that video's transcript.")


# ---------- Sidebar: API key ----------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    st.caption("Get one at platform.openai.com/api-keys. Not stored anywhere.")
    k = st.slider("Chunks to retrieve (k)", 2, 8, 4)
    st.divider()
    st.markdown("Built with LangChain + FAISS + OpenAI")


# ---------- Helpers ----------
def extract_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL or a raw video ID and returns the video ID."""
    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id
    return url_or_id  # fall back, let the API raise a clear error


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_chain(vector_store, api_key: str, k: int):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=["context", "question"],
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel(
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    )
    parser = StrOutputParser()
    return parallel_chain | prompt | llm | parser, retriever


# ---------- Session state ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "history" not in st.session_state:
    st.session_state.history = []


# ---------- Step 1: Load video ----------
video_input = st.text_input("YouTube video URL or ID", placeholder="https://www.youtube.com/watch?v=...")
load_clicked = st.button("Load video", type="primary", use_container_width=True)

if load_clicked:
    if not api_key:
        st.error("Add your OpenAI API key in the sidebar first.")
    elif not video_input:
        st.error("Paste a YouTube URL or video ID.")
    else:
        video_id = extract_video_id(video_input)
        with st.spinner("Fetching transcript and building the index..."):
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.fetch(video_id, languages=["en"])
                transcript = " ".join(snippet.text for snippet in transcript_list)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                embeddings = get_embeddings()
                vector_store = FAISS.from_documents(chunks, embeddings)

                st.session_state.vector_store = vector_store
                st.session_state.video_id = video_id
                st.session_state.history = []
                st.success(f"Indexed {len(chunks)} chunks from the transcript. Ask away below!")
            except TranscriptsDisabled:
                st.error("Captions are disabled for this video.")
            except NoTranscriptFound:
                st.error("No English transcript found for this video.")
            except Exception as e:
                st.error(f"Couldn't process this video: {e}")


# ---------- Preview ----------
if st.session_state.video_id:
    st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")


# ---------- Step 2: Ask questions ----------
if st.session_state.vector_store is not None:
    st.divider()
    question = st.text_input("Ask a question about this video")
    ask_clicked = st.button("Ask", use_container_width=True)

    if ask_clicked and question:
        if not api_key:
            st.error("Add your OpenAI API key in the sidebar first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    chain, retriever = build_chain(st.session_state.vector_store, api_key, k)
                    answer = chain.invoke(question)
                    st.session_state.history.append((question, answer))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q: {q}**")
        st.write(a)
        st.markdown("---")
else:
    st.info("Load a video above to start asking questions.")

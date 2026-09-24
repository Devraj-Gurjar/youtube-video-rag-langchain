# youtube-video-rag-langchain
A Retrieval-Augmented Generation (RAG) system that answers questions about YouTube videos using their transcripts. Built with LangChain, FAISS, and OpenAI, with a Streamlit web interface for live use.

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=6A0DAD,4B0082,7B2FBE,5C16C3&height=200&section=header&text=YouTube%20Video%20RAG&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=Ask%20Any%20YouTube%20Video%20a%20Question%20%7C%20LangChain%20%2B%20FAISS%20%2B%20OpenAI&descAlignY=58&descAlign=50" />

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=20&duration=3000&pause=1000&color=B5A0F6&center=true&vCenter=true&multiline=false&width=700&lines=Turning+YouTube+Transcripts+into+a+Searchable+Knowledge+Base;Retrieval-Augmented+Generation+with+LangChain+%2B+FAISS;Answers+Grounded+in+Context+-+Not+Hallucinated)](https://git.io/typing-svg)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://devraj-gurjar-youtube-video-rag-langchain-app-eb1a3l.streamlit.app/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-005571?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)

[![Stars](https://img.shields.io/github/stars/Devraj-Gurjar/youtube-video-rag-langchain?style=for-the-badge&color=6A0DAD)](../../stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/Devraj-Gurjar/youtube-video-rag-langchain?style=for-the-badge&color=7B2FBE)](../../commits/main)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

### 🔗 [**Try it Live**](https://devraj-gurjar-youtube-video-rag-langchain-app-eb1a3l.streamlit.app/)

**[Live Demo](https://devraj-gurjar-youtube-video-rag-langchain-app-eb1a3l.streamlit.app/)** &nbsp;|&nbsp; **[Report Bug](../../issues)** &nbsp;|&nbsp; **[Request Feature](../../issues)**

</div>

<br/>

## Overview

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline that turns any YouTube video into a searchable knowledge base. Paste a link, ask a question in plain English, and get an answer grounded strictly in that video's transcript — no hallucination, no guessing.

**Live app:** https://devraj-gurjar-youtube-video-rag-langchain-app-eb1a3l.streamlit.app/

Available both as a **Jupyter notebook** (to learn the pipeline step by step) and a **Streamlit web app** (to use it live).

<div align="center">
<table>
<tr>
<td align="center"><b>Input</b><br/>Any YouTube video with captions</td>
<td align="center"><b>Retrieval</b><br/>FAISS similarity search</td>
<td align="center"><b>Generation</b><br/><code>gpt-4o-mini</code>, context-grounded</td>
<td align="center"><b>Interface</b><br/>Streamlit web app</td>
</tr>
</table>
</div>

## How It Works

| Stage | What happens |
|---|---|
| 1. Indexing — Ingestion | Fetches the video transcript via `youtube-transcript-api` |
| 2. Indexing — Splitting | Breaks transcript into overlapping chunks with `RecursiveCharacterTextSplitter` |
| 3. Indexing — Embedding | Encodes chunks using `sentence-transformers/all-MiniLM-L6-v2` |
| 4. Retrieval | Finds the top-k most relevant chunks for a given question via FAISS |
| 5. Augmentation | Builds a prompt combining retrieved context + the question |
| 6. Generation | `gpt-4o-mini` answers **only** from the provided context |

The retrieval → prompt → LLM → parser flow is wired together as a single **LangChain Expression Language (LCEL)** chain — the whole pipeline runs with one `chain.invoke(question)` call.

## Tech Stack

<div align="center">

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-005571?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

</div>

## Project Structure

## Getting Started

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
git clone https://github.com/Devraj-Gurjar/youtube-video-rag-langchain.git
cd youtube-video-rag-langchain
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your-openai-api-key-here
```

### Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste your API key in the sidebar (if not using `.env`), enter a YouTube link, click **Load video**, and start asking questions.

### Run the notebook

```bash
jupyter notebook rag_using_langchain.ipynb
```

## Usage Example

```python
question = "Can you summarize the video?"
answer = main_chain.invoke(question)
print(answer)
```

## Deployment

This app is live on **Streamlit Community Cloud**: https://devraj-gurjar-youtube-video-rag-langchain-app-eb1a3l.streamlit.app/

To deploy your own copy:

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Create a new app → this repo → branch `main` → main file `app.py`.
4. Under **Advanced settings → Secrets**, add:
```toml
   OPENAI_API_KEY = "your-key-here"
```
5. Deploy — live at `https://<your-app-name>.streamlit.app`.

## Roadmap

- [ ] Multi-video support with a persistent vector store
- [ ] Source-chunk citations alongside generated answers
- [ ] Swap FAISS for a hosted vector database (Pinecone / Chroma) at scale
- [ ] Conversation memory for follow-up questions

## Security

API keys are never hardcoded. Local development uses a `.env` file, and deployment uses Streamlit Cloud's encrypted secrets manager — both excluded from version control via `.gitignore`.

## License

Distributed under the MIT License.

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=6A0DAD,4B0082,7B2FBE,5C16C3&height=120&section=footer" />

**Devraj Patel** — B.S. Data Science and Applications, IIT Madras

[![GitHub](https://img.shields.io/badge/GitHub-Devraj--Gurjar-181717?style=for-the-badge&logo=github)](https://github.com/Devraj-Gurjar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/devraj-patel-9257013a0)

</div>

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = Path(__file__).parent / "Medical_book.pdf"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

st.set_page_config(
    page_title="Medical Encyclopedia AI",
    page_icon="🩺",
    layout="centered",
)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_pinecone_index():
    from pinecone import Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")
    pc = Pinecone(api_key=api_key)
    return pc


def get_vector_store():
    from langchain_pinecone import PineconeVectorStore
    embeddings = get_embeddings()
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )


def build_index():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    if not PDF_PATH.exists():
        raise RuntimeError("Medical_book.pdf was not found in the project folder.")

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    pc = Pinecone(api_key=api_key)
    existing = pc.list_indexes().names()
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    return len(documents), len(chunks)


def get_rag_chain():
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    system_prompt = (
        "You are a medical encyclopedia assistant. "
        "Answer using only the retrieved context from the supplied medical encyclopedia. "
        "If the context does not contain the answer, say that the information was not found "
        "in the provided encyclopedia. Do not invent medical facts. "
        "Keep answers clear and concise. Do not diagnose the user or prescribe treatment. "
        "For urgent or personal medical concerns, advise the user to consult a qualified healthcare professional."
        "\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


st.markdown(
    "<h1 style='text-align:center;'>🩺 Medical Encyclopedia AI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Ask questions and get answers grounded in the provided medical encyclopedia.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project Setup")
    st.write(f"Index: `{INDEX_NAME}`")
    st.write(f"Embedding: `{EMBEDDING_MODEL}`")

    if st.button("Build / Update Pinecone Index", use_container_width=True):
        with st.spinner("Reading the medical book and building the index..."):
            try:
                pages, chunks = build_index()
                st.success(f"Done. Processed {pages} pages into {chunks} chunks.")
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.caption(
        "This application is an educational information tool. It does not replace a doctor, diagnosis, or professional medical advice."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something from the medical encyclopedia...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the encyclopedia..."):
            try:
                chain = get_rag_chain()
                answer = chain.invoke(question)
                st.markdown(answer)
                st.caption("Source: retrieved content from the provided medical encyclopedia.")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                error = str(exc)
                st.error(error)
                st.info(
                    "Check your GROQ_API_KEY and PINECONE_API_KEY in .env. "
                    "If the Pinecone index has not been created/populated yet, use the sidebar button first."
                )

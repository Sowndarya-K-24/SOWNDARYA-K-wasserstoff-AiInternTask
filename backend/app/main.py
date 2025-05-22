import os
os.environ["HOME"] = "/tmp"
os.environ["STREAMLIT_HOME"] = "/tmp"

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import tempfile
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pdf2image import convert_from_path
import shutil 


# --- Check if collection exists before creating ---

from chromadb import PersistentClient
from chromadb.errors import NotFoundError, UniqueConstraintError, InternalError

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

client = PersistentClient(path="/app/chroma_storage")
collection_name = "documents"

from chromadb.errors import UniqueConstraintError

try:
    collection = client.get_collection(name=collection_name)
except NotFoundError:
    try:
        collection = client.create_collection(name=collection_name)
    except (UniqueConstraintError, InternalError):
        collection = client.get_collection(name=collection_name)

embedder = SentenceTransformer(EMBEDDING_MODEL)

st.set_page_config(layout="wide")
st.title("📄 Document Research & Theme Identification Chatbot")

# -- Show uploaded documents --
st.sidebar.header("Uploaded Documents")
uploaded_docs = set([m['doc_name'] for m in collection.get()['metadatas']])
if uploaded_docs:
    st.sidebar.write("📄 **Available Documents:**")
    for doc in sorted(uploaded_docs):
        st.sidebar.markdown(f"- {doc}")
else:
    st.sidebar.info("No documents uploaded yet.")

# -- Add filters --
st.sidebar.header("Filters")
selected_doc = st.sidebar.selectbox("Filter by Document", ["All"] + sorted(uploaded_docs))
selected_type = st.sidebar.selectbox("Filter by Document Type", ["All", "PDF", "Scanned"])

# --- Utility Functions ---

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        full_text += text + "\n"
    return full_text

def extract_text_from_scanned_pdf(uploaded_file):
    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(uploaded_file.name, output_folder=path)
        text = ""
        for i, image in enumerate(images):
            text += pytesseract.image_to_string(image)
    return text

def embed_and_store(text, doc_name):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks).tolist()
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], embeddings=[embeddings[i]], metadatas=[{"doc_name": doc_name, "chunk_id": i}], ids=[f"{doc_name}_{i}"])

def query_documents(query):
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    return results

# --- Upload Section ---

st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Choose PDFs or scanned PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.sidebar.write(f"Processing: {file.name}")
        if "scanned" in file.name.lower():
            text = extract_text_from_scanned_pdf(file)
        else:
            text = extract_text_from_pdf(file)
        embed_and_store(text, file.name)
        st.sidebar.success(f"Uploaded & Indexed: {file.name}")

# --- Query Section ---
st.header("Ask a Question Across All Uploaded Documents")
query = st.text_input("Enter your query:", placeholder="e.g., What are the key regulations mentioned?")

filtered_results = []

if query:
    results = query_documents(query)

     # -- Apply filters --
    filtered_results = []
    for result, metadata, doc_id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
        if selected_doc != "All" and metadata['doc_name'] != selected_doc:
            continue
        if selected_type != "All" and metadata.get("document_type", "PDF") != selected_type:
            continue
        filtered_results.append((result, metadata, doc_id))

    st.subheader("📄 Relevant Document Excerpts:")
    for i in range(len(results["documents"][0])):
        doc_id = results["ids"][0][i]
        doc_text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        st.markdown(f"**{metadata['doc_name']}** - Chunk {metadata['chunk_id']} (ID: {doc_id})")
        st.write(doc_text)
        st.markdown("---")

elif not filtered_results:
    st.warning("No matching results found based on filters.")

# -- Tabular Output with Citations --

def extract_citation(text):
     
     return "Location unknown"

if filtered_results:
    table_data = []
    for result, metadata, doc_id in filtered_results:
        table_data.append({
            "Document ID": metadata.get("doc_name", "N/A"),
            "Extracted Answer": result[:300] + ("..." if len(result) > 300 else ""),
            "Citation": extract_citation(result)
        })
    df = pd.DataFrame(table_data)
    st.subheader("📊 Tabular View with Citations")
    st.dataframe(df, use_container_width=True)

# --- Theme Identification ---

if query:
    st.subheader("🧠 Synthesized Theme (Experimental)")
    all_texts = results['documents'][0]
    theme = pipeline("summarization")("\n".join(all_texts[:3]))[0]['summary_text']
    st.success(theme)
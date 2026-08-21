import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ==========================================
# Configuration
# ==========================================

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

# ==========================================
# Load PDF Files
# ==========================================

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No PDF files found inside '{data_path}' folder."
        )

    print(f"Loaded {len(documents)} pages.")

    return documents


# ==========================================
# Create Text Chunks
# ==========================================

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ==========================================
# Embedding Model
# ==========================================

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==========================================
# Create FAISS Vector Store
# ==========================================

def create_vector_store():

    print("Loading PDFs...")

    documents = load_pdf_files(DATA_PATH)

    print("Creating chunks...")

    chunks = create_chunks(documents)

    print("Loading embedding model...")

    embedding_model = get_embedding_model()

    print("Creating FAISS vector database...")

    db = FAISS.from_documents(
        chunks,
        embedding_model
    )

    os.makedirs(DB_FAISS_PATH, exist_ok=True)

    db.save_local(DB_FAISS_PATH)

    print("===================================")
    print("✅ Vector Store Created Successfully!")
    print(f"Saved to: {DB_FAISS_PATH}")
    print("===================================")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    create_vector_store()

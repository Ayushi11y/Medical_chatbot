from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import

import os

# Paths
DATA_FOLDER = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Load PDFs
loader = DirectoryLoader(DATA_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

if not documents:
    raise ValueError("❌ No PDF documents found in the 'data/' folder.")

# Step 2: Embed and store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local(DB_FAISS_PATH)

print("✅ PDF ingestion complete. FAISS vectorstore updated.")



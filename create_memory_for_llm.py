from langchain.vectorstores import FAISS  # or Chroma, etc.
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.document_loaders import TextLoader  # or JSONLoader, PDFLoader, etc.
import os

# Step 1: Set Groq API Key
os.environ["GROQ_API_KEY"] = "your_groq_api_key"

# Step 2: Load or create your vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # or any compatible model

# Example: Load existing FAISS vectorstore from disk
vectorstore = FAISS.load_local("vectorstore_path", embeddings=embedding_model)

# Step 3: Define the LLM (Groq)
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-8b-8192"  # or "mixtral-8x7b-32768"
)

# Step 4: Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Step 5: Get user query and run chain
user_query = "What type of data we have"

# Optional Debug: Print what documents are being retrieved
retrieved_docs = vectorstore.similarity_search(user_query, k=3)
print("📄 Retrieved Documents:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n--- Document {i} ---\n{doc.page_content}")

# Step 6: Get response from chain
response = qa_chain.invoke({'query': user_query})

# Step 7: Print the response
print("\n🤖 Answer:")
print(response["result"])

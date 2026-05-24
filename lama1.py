import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Path to your PDF document
PDF_FILENAME = "Unit_IV_Lasers (1).pdf"

if not os.path.exists(PDF_FILENAME):
    print(f"❌ Error: '{PDF_FILENAME}' not found in this folder.")
    exit()

print("🔄 Loading and parsing document...")
loader = PyPDFLoader(PDF_FILENAME)
documents = loader.load()

print("✂️ Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
chunks = text_splitter.split_documents(documents)

print("🧬 Generating vector embeddings (nomic-embed-text)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

print("📦 Building local Vector Database index...")
vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

print("🤖 Connecting to local LLaMA 3 engine...")
llm = Ollama(model="llama3", base_url="http://localhost:11434")

# Modern LCEL Prompt Setup
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Modern RAG Chain Definition (Bypasses legacy RetrievalQA)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Run Query ---
user_query = "tell me components in ruby laser?"
print(f"\n❓ Question: {user_query}\n🧠 Generating response...")

response = rag_chain.invoke(user_query)
print("\n✨ Local LLaMA 3 Answer:")
print(response)
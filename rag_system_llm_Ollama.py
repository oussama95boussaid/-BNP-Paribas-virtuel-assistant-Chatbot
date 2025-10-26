# RAG System - BNP Paribas Knowledge Base

# Stack:
# - LangChain for orchestration
# - Google Gemini API for embeddings & generation
# - ChromaDB for vector storage
# - Multilingual support (French & English)



# %%
# Install required packages - HuggingFace Embeddings + Ollama
# !pip install langchain==0.3.17 langchain-community==0.3.16 langchain-openai==0.2.14 chromadb==0.5.23 sentence-transformers==3.3.1 langchain-huggingface==0.1.2 python-dotenv
# pip install "langchain-core>=0.3.33,<0.4.0"

# %%
import json
import os
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import warnings

# Prefer the up-to-date `langchain-ollama` package. Fall back to the older
# `langchain_community.llms.Ollama` if the new package isn't installed.
try:
    # Newer package and class
    from langchain_ollama import OllamaLLM as _OllamaClass
except Exception:
    # Fallback for older langchain-community shim
    try:
        from langchain_community.llms import Ollama as _OllamaClass
        warnings.warn(
            "Using legacy `langchain_community.llms.Ollama`."
            " Install `langchain-ollama` and update imports to"
            " `from langchain_ollama import OllamaLLM` to remove this warning.",
            DeprecationWarning,
        )
    except Exception:
        raise







# %%

JSON_FILE = "bnp_rag_docs/rag_documents.json"
CHROMA_DB_DIR = "chroma_db"

# Embedding Model Configuration (HuggingFace)
# Best multilingual model for French & English
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Alternative models:
# "sentence-transformers/distiluse-base-multilingual-cased-v1"
# "intfloat/multilingual-e5-large" (better quality, slower)

# LLM Configuration (Ollama)
LLM_MODEL = "mistral"  # or "llama3.2", "gemma2", "qwen2.5"

# Chunking Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Retrieval Configuration
TOP_K_RESULTS = 4  # Number of relevant documents to retrieve

print("Configuration:")
print(f"  Embedding Model: {EMBEDDING_MODEL} (HuggingFace)")
print(f"  LLM Model: {LLM_MODEL} (Ollama)")
print(f"  Chunk Size: {CHUNK_SIZE}")
print(f"  Top K Results: {TOP_K_RESULTS}")



# %%
def load_documents_from_json(json_path: str) -> List[Document]:
    """Load scraped documents from JSON file"""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for doc in data:
        # Create LangChain Document with content and metadata
        langchain_doc = Document(
            page_content=doc['content'],
            metadata={
                'id': doc['id'],
                'category': doc['category'],
                'title': doc['title'],
                'url': doc['url'],
                'description': doc['description'],
                'word_count': doc['word_count'],
                'scraped_at': doc['scraped_at']
            }
        )
        documents.append(langchain_doc)
    
    return documents

# Load documents
print(f"Loading documents from {JSON_FILE}...")
documents = load_documents_from_json(JSON_FILE)
print(f"✓ Loaded {len(documents)} documents")

# Display sample
print(f"\nSample document:")
print(f"  Category: {documents[0].metadata['category']}")
print(f"  Title: {documents[0].metadata['title']}")
print(f"  Content length: {len(documents[0].page_content)} chars")



# %%
# Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Split documents
print("Splitting documents into chunks...")
splits = text_splitter.split_documents(documents)
print(f"✓ Created {len(splits)} chunks from {len(documents)} documents")

# Statistics
total_chars = sum(len(split.page_content) for split in splits)
avg_chunk_size = total_chars / len(splits)
print(f"  Average chunk size: {avg_chunk_size:.0f} characters")



# %%
# Initialize HuggingFace embeddings (FREE, runs locally)
print("Initializing HuggingFace embeddings model...")
print("⏳ First run will download the model (~100MB)...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}  # Important for better retrieval
)
print("✓ Embeddings model initialized (running locally)")

# Test embeddings
test_text = "Test de l'embedding en français et anglais"
test_embedding = embeddings.embed_query(test_text)
print(f"  Embedding dimension: {len(test_embedding)}")



# %%
# Create Vector Store with ChromaDB
print(f"Creating ChromaDB vector store in '{CHROMA_DB_DIR}'...")

# Remove existing database if you want to recreate
import shutil
if Path(CHROMA_DB_DIR).exists():
    shutil.rmtree(CHROMA_DB_DIR)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_DB_DIR,
    collection_name="bnp_paribas_docs"
)

print(f"✓ Vector store created with {len(splits)} embeddings")

# %%
# Test similarity search
test_query = "Quelles sont les cartes bancaires disponibles ?"
print(f"Test query: '{test_query}'")
print("\nRetrieving relevant documents...\n")

results = vectorstore.similarity_search(test_query, k=3)

for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"  Category: {doc.metadata['category']}")
    print(f"  Title: {doc.metadata['title']}")
    print(f"  Content preview: {doc.page_content[:200]}...")
    print()



# %%
# Custom prompt template for French/English responses
prompt_template = """Vous êtes un assistant bancaire expert de BNP Paribas. Utilisez les informations suivantes pour répondre à la question de manière précise et professionnelle.

You are an expert banking assistant for BNP Paribas. Use the following information to answer the question accurately and professionally.

IMPORTANT:
- Répondez dans la langue de la question (français ou anglais)
- Answer in the language of the question (French or English)
- Si l'information n'est pas dans le contexte, dites-le clairement
- If information is not in the context, say so clearly
- Citez les sources pertinentes (catégorie, URL)
- Cite relevant sources (category, URL)

Context:
{context}

Question: {question}

Réponse détaillée / Detailed answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

print("✓ Custom bilingual prompt created")



# %%
# Initialize Ollama LLM (runs locally)
print(f"Initializing Ollama with {LLM_MODEL}...")
llm = _OllamaClass(
    model=LLM_MODEL,
    temperature=0.3,  # Lower = more focused, higher = more creative
)
print("✓ Ollama LLM initialized (running locally)")



# %%
# Create RAG Chain using LCEL (LangChain Expression Language)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K_RESULTS}
)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

print("✓ RAG chain created successfully!")



# %%
# Query Function
def query_rag(question: str, show_sources: bool = True) -> Dict:
    """
    Query the RAG system
    
    Args:
        question: User question in French or English
        show_sources: Whether to display source documents
    
    Returns:
        Dictionary with answer and sources
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # Get answer using LCEL chain
    answer = rag_chain.invoke(question)
    
    # Get source documents separately
    sources = retriever.invoke(question)
    
    print("Answer:")
    print(answer)
    print()
    
    if show_sources:
        print(f"\n{'─'*60}")
        print(f"Sources ({len(sources)} documents):")
        print(f"{'─'*60}")
        for i, doc in enumerate(sources, 1):
            print(f"\n{i}. {doc.metadata['category']}")
            print(f"   Title: {doc.metadata['title']}")
            print(f"   URL: {doc.metadata['url']}")
            print(f"   Preview: {doc.page_content[:150]}...")
    
    return {
        'question': question,
        'answer': answer,
        'sources': [
            {
                'category': doc.metadata['category'],
                'title': doc.metadata['title'],
                'url': doc.metadata['url']
            }
            for doc in sources
        ]
    }

print("✓ Query function ready")



# %%
# Test query in French
result1 = query_rag("Quelles sont les différentes cartes Visa disponibles et leurs avantages ?")

# %%
# Test query in English
result2 = query_rag("What are the savings account options available?")

 # %%
# Test query about specific topic
# result3 = query_rag("Comment puis-je financer un projet immobilier ?")

# %%
# Test query about security
# result4 = query_rag("What security measures does BNP Paribas have?")



# %%
# Interactive Query Interface
def interactive_chat():
    """Interactive chat interface"""
    print("\n" + "="*60)
    print("BNP PARIBAS RAG ASSISTANT")
    print("="*60)
    print("Ask questions in French or English")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nAu revoir! / Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            query_rag(question, show_sources=True)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

# Uncomment to run interactive chat
interactive_chat()

# %% [markdown]
# ## 16. Save RAG Configuration

# %%
# Save configuration for future use
config = {
    "embedding_model": EMBEDDING_MODEL,
    "llm_model": LLM_MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "top_k_results": TOP_K_RESULTS,
    "chroma_db_dir": CHROMA_DB_DIR,
    "total_documents": len(documents),
    "total_chunks": len(splits)
}

config_path = Path(CHROMA_DB_DIR) / "config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Configuration saved to {config_path}")

# %% [markdown]
# ## 17. Summary & Next Steps

# %%
print("\n" + "="*60)
print("RAG SYSTEM SETUP COMPLETE! 🎉")
print("="*60)
print(f"✓ Documents loaded: {len(documents)}")
print(f"✓ Chunks created: {len(splits)}")
print(f"✓ Vector store: {CHROMA_DB_DIR}")
print(f"✓ Model: {LLM_MODEL}")
print(f"✓ Multilingual: French & English")
print("="*60)
print("\nYou can now:")
print("1. Use query_rag() function to ask questions")
print("2. Run interactive_chat() for conversational interface")
print("3. Integrate into your application")
print("4. Deploy as API or chatbot")
print("="*60)

# %% [markdown]
# ## Example Usage
# 
# ```python
# # Simple query
result = query_rag("Quels sont les avantages de la carte Visa Premier?")
# 
# # Get answer only
# answer = result['answer']
# 
# # Get sources
# sources = result['sources']
# 
# # Interactive mode
# interactive_chat()
# ```
# %%

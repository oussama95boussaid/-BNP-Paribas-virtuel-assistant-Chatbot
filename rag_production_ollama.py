"""
Production RAG System - BNP Paribas Assistant
HuggingFace Embeddings + Ollama + FastAPI
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM as _OllamaClass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# ============================================================================
# RAG ASSISTANT CLASS
# ============================================================================

class BNPAssistant:
    """Production RAG Assistant - loads existing vector DB"""
    
    def __init__(self, config_path: str = "config/config_ollama.json"):
        """Initialize assistant by loading existing configuration and DB"""
        
        print("🚀 Initializing BNP Paribas Assistant...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize embeddings 
        print("📦 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model'],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embeddings loaded")
        
        # Load existing ChromaDB
        print("💾 Loading vector database...")
        self.vectorstore = Chroma(
            persist_directory=self.config['chroma_db_dir'],
            embedding_function=self.embeddings,
            collection_name="bnp_paribas_docs"
        )
        print(f"✓ Vector DB loaded ({self.config['total_chunks']} chunks)")
        
        # Initialize LLM
        print(f"🤖 Initializing {self.config['llm_model']}...")
        self.llm = _OllamaClass(
            model=self.config['llm_model'],
            temperature=0.3
        )
        print("✓ LLM ready")
        
        # Create prompt template
        self.prompt = self._create_prompt()
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config['top_k_results']}
        )
        
        # Create RAG chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ Assistant ready!\n")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                "Please run the setup notebook first to create the vector database."
            )
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"✓ Config loaded: {config['total_documents']} docs, {config['total_chunks']} chunks")
        return config
    
    def _create_prompt(self) -> PromptTemplate:
        """Create bilingual prompt template"""
        template = """Vous êtes un assistant bancaire expert de BNP Paribas. Utilisez les informations suivantes pour répondre à la question de manière précise et professionnelle.

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
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str, include_sources: bool = True) -> Dict:
        """
        Ask a question to the assistant
        
        Args:
            question: Question in French or English
            include_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and sources
        """ 
        # Get answer using LCEL chain
        answer = self.rag_chain.invoke(question)
        
        # Get source documents separately
        sources = self.retriever.invoke(question)
        
        response = {
            'question': question,
            'answer': answer
        }
        
        if include_sources:
            response['sources'] = [
                {
                    'category': doc.metadata['category'],
                    'title': doc.metadata['title'],
                    'url': doc.metadata['url']
                }
                for doc in sources
            ]
        
        return response
    

    
    


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="BNP Paribas RAG Assistant API",
    description="AI-powered banking assistant using RAG with Ollama (100% Free & Local)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize assistant (loads once at startup)
assistant: Optional[BNPAssistant] = None


# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question in French or English", min_length=1)
    include_sources: bool = Field(True, description="Include source documents in response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Quelles sont les cartes bancaires disponibles ?",
                "include_sources": True
            }
        }


class Source(BaseModel):
    category: str
    title: str
    url: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[Source]] = None



class HealthResponse(BaseModel):
    status: str
    message: str
    config: Optional[Dict] = None


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on startup"""
    global assistant
    try:
        assistant = BNPAssistant()
        print("✅ FastAPI server ready!")
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API information"""
    return {
        "status": "online",
        "message": "BNP Paribas RAG Assistant API is running (Ollama - 100% Free)",
        "config": assistant.config if assistant else None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    return {
        "status": "healthy",
        "message": "All systems operational",
        "config": {
            "total_documents": assistant.config['total_documents'],
            "total_chunks": assistant.config['total_chunks'],
            "llm_model": assistant.config['llm_model'],
            "embedding_model": assistant.config['embedding_model']
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question to the assistant
    
    - **question**: Your question in French or English
    - **include_sources**: Whether to include source documents (default: true)
    """
    if assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        result = assistant.query(
            question=request.question,
            include_sources=request.include_sources
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")



# ============================================================================
# STANDALONE SCRIPT MODE
# ============================================================================

def run_cli():
    """Run in CLI mode"""
    print("\n" + "="*60)
    print("BNP PARIBAS ASSISTANT - CLI MODE")
    print("="*60 + "\n")
    
    # Initialize assistant
    assistant = BNPAssistant()
    
    # Example 1: Single query in French
    print("\n" + "="*60)
    print("EXAMPLE 1: French Query")
    print("="*60)
    result1 = assistant.query("Quelles sont les cartes bancaires disponibles ?")
    print(f"\n❓ {result1['question']}")
    print(f"💬 {result1['answer']}")
    if result1.get('sources'):
        print(f"\n📚 Sources:")
        for i, source in enumerate(result1['sources'], 1):
            print(f"  {i}. {source['category']} - {source['url']}")
    
    # Example 2: Single query in English
    print("\n" + "="*60)
    print("EXAMPLE 2: English Query")
    print("="*60)
    result2 = assistant.query("What savings accounts do you offer?")
    print(f"\n❓ {result2['question']}")
    print(f"💬 {result2['answer']}")
    if result2.get('sources'):
        print(f"\n📚 Sources:")
        for i, source in enumerate(result2['sources'], 1):
            print(f"  {i}. {source['category']} - {source['url']}")
    
    # Example 3: Batch queries
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Queries")
    print("="*60)
    questions = [
        "Quels sont les livrets d'épargne disponibles ?",
        "How can I finance a real estate project?"
    ]
    batch_results = assistant.batch_query(questions)
    for i, result in enumerate(batch_results, 1):
        print(f"\n{i}. {result['question']}")
        print(f"   Answer: {result['answer'][:150]}...")
    
    print("\n" + "="*60)
    print("✅ CLI examples completed!")
    print("="*60)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run FastAPI server"""
    print("\n" + "="*60)
    print("STARTING FASTAPI SERVER")
    print("="*60)
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    print(f"Alternative docs: http://{host}:{port}/redoc")
    print("="*60 + "\n")
    
    uvicorn.run(app, host=host, port=port)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "server":
            # Run FastAPI server
            run_server()
        elif sys.argv[1] == "chat":
            # Run interactive chat
            assistant = BNPAssistant()
            assistant.chat()
        else:
            print("Usage:")
            print("  python rag_production.py           # Run CLI examples")
            print("  python rag_production.py server    # Start FastAPI server")
            print("  python rag_production.py chat      # Interactive chat mode")
    else:
        # Run CLI examples
        run_cli()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("To start FastAPI server:")
        print("  python rag_production.py server")
        print("\nTo start interactive chat:")
        print("  python rag_production.py chat")
        print("\nOr manually:")
        print("  uvicorn rag_production:app --reload")
        print("="*60)
        
        
        
        
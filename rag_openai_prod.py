"""
Production RAG System - BNP Paribas Assistant
HuggingFace Embeddings + OpenAI + FastAPI
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# ============================================================================
# RAG ASSISTANT CLASS
# ============================================================================

class BNPAssistant:
    """Production RAG Assistant with OpenAI"""
    
    def __init__(self, config_path: str = "config/config_gpt.json"):
        """Initialize assistant by loading existing configuration and DB"""
        
        print("Initializing BNP Paribas Assistant (OpenAI)...")
        
        # Load environment variables
        load_dotenv()
        
        # Verify OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found! Please set it in .env file or environment."
            )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize embeddings
        print("Loading HuggingFace embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model'],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embeddings loaded")
        
        # Load existing ChromaDB
        print("Loading vector database...")
        self.vectorstore = Chroma(
            persist_directory=self.config['chroma_db_dir'],
            embedding_function=self.embeddings,
            collection_name="bnp_paribas_docs"
        )
        print(f"✓ Vector DB loaded ({self.config['total_chunks']} chunks)")
        
        # Initialize OpenAI LLM
        print(f"Initializing OpenAI {self.config['llm_model']}...")
        self.llm = ChatOpenAI(
            model=self.config['llm_model'],
            temperature=0.3,
            max_tokens=1000
        )
        print("✓ OpenAI LLM ready")
        
        # Create prompt template
        self.prompt = self._create_prompt()
        
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
        # Get answer
        result = self.qa_chain.invoke({"query": question})
        
        answer = result['result']
        sources = result['source_documents']
        
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
    
    # def batch_query(self, questions: List[str]) -> List[Dict]:
    #     """Process multiple questions"""
    #     results = []
    #     for q in questions:
    #         result = self.query(q, include_sources=False)
    #         results.append(result)
    #     return results


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="BNP Paribas RAG Assistant API",
    description="AI-powered banking assistant using RAG (Retrieval-Augmented Generation)",
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


# class BatchQueryRequest(BaseModel):
#     questions: List[str] = Field(..., description="List of questions", min_items=1, max_items=10)
    
#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "questions": [
#                     "Quelles cartes bancaires proposez-vous ?",
#                     "What savings accounts are available?"
#                 ]
#             }
#         }


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
        "message": "BNP Paribas RAG Assistant API is running",
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
            "llm_model": assistant.config['llm_model']
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


# @app.post("/batch-query", response_model=List[QueryResponse])
# async def batch_query(request: BatchQueryRequest):
#     """
#     Process multiple questions at once
    
#     - **questions**: List of questions (max 10)
#     """
#     if assistant is None:
#         raise HTTPException(status_code=503, detail="Assistant not initialized")
    
#     try:
#         results = assistant.batch_query(request.questions)
#         return results
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing batch query: {str(e)}")


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
    
    # Example queries
    print("\n" + "="*60)
    print("EXAMPLE 1: French Query")
    print("="*60)
    result1 = assistant.query("Quelles sont les cartes bancaires disponibles ?")
    print(f"\n❓ {result1['question']}")
    print(f"💬 {result1['answer']}")
    print(f"📚 Sources: {len(result1['sources'])}")
    
    print("\n" + "="*60)
    print("EXAMPLE 2: English Query")
    print("="*60)
    result2 = assistant.query("What savings accounts do you offer?")
    print(f"\n❓ {result2['question']}")
    print(f"💬 {result2['answer']}")
    print(f"📚 Sources: {len(result2['sources'])}")
    
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
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run FastAPI server
        run_server()
    else:
        # Run CLI examples
        run_cli()
        
        print("\n" + "="*60)
        print("To start FastAPI server, run:")
        print("  python rag_openai_prod.py server")
        print("\nOr manually:")
        print("  uvicorn rag_openai_prod:app --reload")
        print("="*60)

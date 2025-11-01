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
        # Create a retriever from the vectorstore
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.get('top_k_results', 3)}
        )

        # Build the LCEL RAG chain
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

        # Retrieve source documents separately for attribution
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
      

app = FastAPI(
    title="BNP Paribas RAG API",
    description="Banking Assistant API",
    version="1.0.0"
)

# CORS - Only allow your frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Development
        "https://chat.vercel.app",  # Production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

assistant: Optional[BNPAssistant] = None

# Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    include_sources: bool = Field(default=True)

class Source(BaseModel):
    category: str
    title: str
    url: str

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = None

class HealthResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int

# Endpoints - 
@app.on_event("startup")
async def startup_event():
    global assistant
    try:
        assistant = BNPAssistant()
        print("✅ FastAPI ready!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check - verify backend is running"""
    if assistant is None:
        raise HTTPException(status_code=503, detail="Not initialized")
    
    return HealthResponse(
        status="healthy",
        total_documents=assistant.config['total_documents'],
        total_chunks=assistant.config['total_chunks']
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main endpoint - answer questions"""
    if assistant is None:
        raise HTTPException(status_code=503, detail="Not initialized")
    
    try:
        result = assistant.query(
            question=request.question,
            include_sources=request.include_sources
        )
        return QueryResponse(
            answer=result['answer'],
            sources=result.get('sources')
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
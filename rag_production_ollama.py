"""
Production RAG System - BNP Paribas Assistant
"""

import json
from pathlib import Path
from typing import Dict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM as _OllamaClass



class BNPAssistant:
    """Production RAG Assistant - loads existing vector DB"""
    
    def __init__(self, config_path: str = "chroma_db/config.json"):
        """Initialize assistant by loading existing configuration and DB"""
        
        print("🚀 Initializing BNP Paribas Assistant...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize embeddings 
        print("Loading embeddings model...")
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
        
        # Initialize LLM
        print(f"Initializing {self.config['llm_model']}...")
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
    
    def query(self, question: str, show_sources: bool = True) -> Dict:
        """
        Ask a question to the assistant
        
        Args:
            question: Question in French or English
            show_sources: Whether to display source documents
            
        Returns:
            Dictionary with answer and sources
        """
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}\n")
        
        # Get answer using LCEL chain
        answer = self.rag_chain.invoke(question)
        
        # Get source documents separately for display
        sources = self.retriever.invoke(question)
        
        print("💬 Answer:")
        print(answer)
        print()
        
        if show_sources:
            print(f"{'─'*60}")
            print(f"📚 Sources ({len(sources)} documents):")
            print(f"{'─'*60}")
            for i, doc in enumerate(sources, 1):
                print(f"\n{i}. {doc.metadata['category']}")
                print(f"   📄 {doc.metadata['title']}")
                print(f"   🔗 {doc.metadata['url']}")
        
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
    
    def chat(self):
        """Interactive chat interface"""
        print("\n" + "="*60)
        print("💬 BNP PARIBAS ASSISTANT - CHAT MODE")
        print("="*60)
        print("Ask questions in French or English")
        print("Commands: 'quit', 'exit', 'q' to stop")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Au revoir! / Goodbye!")
                    break
                
                if not question:
                    continue
                
                self.query(question, show_sources=True)
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir! / Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    # def batch_query(self, questions: List[str]) -> List[Dict]:
    #     """Process multiple questions"""
    #     results = []
    #     for q in questions:
    #         result = self.query(q, show_sources=False)
    #         results.append(result)
    #     return results


# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Initialize assistant (loads existing DB)
    assistant = BNPAssistant()
    
    # Example 1: Single query
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Query")
    print("="*60)
    result = assistant.query(
        "Quelles sont les cartes bancaires disponibles ?",
        show_sources=True
    )
    
    
    
    # Example 3: Interactive chat (uncomment to use)
    # assistant.chat()
    
    print("\n" + "="*60)
    print("✅ Script completed successfully!")
    print("="*60)
    print("\nTo start interactive chat, uncomment:")
    print("  assistant.chat()")

# BNP Paribas Virtual Assistant Chatbot 🏦💬

A production-ready RAG (Retrieval-Augmented Generation) system that provides intelligent answers to customer questions about BNP Paribas banking products and services.

## 🎯 Project Objectives

- **Intelligent Banking Assistant**: Provide accurate, context-aware answers about BNP Paribas products, services, and procedures
- **Multilingual Support**: Answer questions in both French and English
- **Source Attribution**: Cite official BNP Paribas documentation for transparency
- **Local-First Architecture**: Run entirely on your machine using open-source models (Ollama + HuggingFace)
- **Production Ready**: Clean API, error handling, and easy deployment

## 🛠️ Tech Stack

- **LangChain 1.x** - RAG orchestration with LCEL (LangChain Expression Language)
- **Ollama** - Local LLM inference (Mistral, Llama, Phi, etc.)
- **HuggingFace Embeddings** - Multilingual sentence embeddings
- **ChromaDB** - Vector database for semantic search
- **Python 3.12+** - Modern Python with type hints

## 📋 Prerequisites

- **Python 3.12+**
- **Ollama** installed and running ([Download here](https://ollama.ai/))
- At least **8GB RAM** (4GB for lighter models like `phi` or `tinyllama`)
- **Git** for cloning the repository

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/oussama95boussaid/-BNP-Paribas-virtuel-assistant-Chatbot.git
cd Backend_server
```

### 2. Create Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1
```

```bash
# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Pull Ollama Model
```bash
# For production (requires ~4.9GB RAM)
ollama pull mistral

# For lighter setup (requires ~2GB RAM)
ollama pull phi
# or
ollama pull tinyllama
```

### 5. Setup Vector Database
Run the setup script to create embeddings from BNP Paribas documents:
```bash
python rag_system_llm_Ollama.py
```

This will:
- Load documents from `bnp_rag_docs/rag_documents.json`
- Create embeddings using HuggingFace
- Store vectors in `chroma_db/` directory
- Save configuration to `chroma_db/config.json`

## 💻 Usage

### Production API

Use the production-ready assistant class:

```python
from rag_production_ollama import BNPAssistant

# Initialize assistant (loads existing vector DB)
assistant = BNPAssistant()

# Ask a question
result = assistant.query(
    "Quelles sont les cartes bancaires disponibles ?",
    show_sources=True
)

print(result['answer'])
print(result['sources'])
```

### Interactive Chat

Run the interactive chat interface:

```python
from rag_production_ollama import BNPAssistant

assistant = BNPAssistant()
assistant.interactive_chat()
```

Then ask questions:
```
You: Quelles sont les cartes Visa disponibles ?
You: What are the savings account options?
You: Comment ouvrir un compte ?
```

### CLI Usage

```bash
# Run production assistant
python rag_production_ollama.py

# Run development/setup script
python rag_system_llm_Ollama.py
```

## 📁 Project Structure

```
Backend_server/
├── rag_production_ollama.py      # Production-ready RAG assistant
├── rag_system_llm_Ollama.py      # Setup script & development
├── bnp_rag_docs/                 # Source documents
│   └── rag_documents.json
├── chroma_db/                     # Vector database (generated)
│   ├── config.json
│   └── ...
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (create this)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## ⚙️ Configuration

### Change LLM Model

Edit the configuration in your script:

```python
# In rag_system_llm_Ollama.py or rag_production_ollama.py
LLM_MODEL = "mistral"  # or "phi", "tinyllama", "llama3.2", "gemma2"
```

Or modify `chroma_db/config.json`:

```json
{
  "llm_model": "phi",
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "chunk_size": 1000,
  "top_k_results": 4
}
```

### Adjust Retrieval Settings

```python
# Number of relevant documents to retrieve
TOP_K_RESULTS = 4  # Increase for more context, decrease for faster responses

# Chunk size for splitting documents
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## 🐛 Troubleshooting

### Memory Issues with Ollama

**Error**: `model requires more system memory (4.9 GiB) than is available`

**Solutions**:
1. Use a smaller model: `ollama pull phi` or `ollama pull tinyllama`
2. Close other applications to free RAM
3. Update config to use the lighter model

### Import Errors

**Error**: `No module named 'langchain.chains'`

**Solution**: Ensure you have LangChain 1.x installed:
```bash
pip install --upgrade langchain langchain-core langchain-community langchain-ollama
```

### Vector DB Not Found

**Error**: `Config file not found: chroma_db/config.json`

**Solution**: Run the setup script first:
```bash
python rag_system_llm_Ollama.py
```

## 🔒 Environment Variables

Create a `.env` file for API keys (if using cloud LLMs):

```bash
# For OpenAI (optional)
OPENAI_API_KEY=your-key-here

# For Google Gemini (optional)
GOOGLE_API_KEY=your-key-here
```

## 📊 Performance Tips

- **Use GPU**: Set `model_kwargs={'device': 'cuda'}` for HuggingFace embeddings if you have NVIDIA GPU
- **Cache embeddings**: The vector DB is persistent, no need to re-embed after first run
- **Batch queries**: Use the API to process multiple questions efficiently
- **Adjust chunk size**: Smaller chunks = faster, larger chunks = more context

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Oussama Boussaid**
- GitHub: [@oussama95boussaid](https://github.com/oussama95boussaid)

## 🙏 Acknowledgments

- BNP Paribas for banking documentation
- LangChain for RAG framework
- Ollama for local LLM inference
- HuggingFace for open-source embeddings

# CTSE Lecture Notes Chatbot - Fernando W.T.R.P.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.x-orange)
![Ollama](https://img.shields.io/badge/Ollama-Gemma:2b-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33.x-red)

An intelligent Retrieval-Augmented Generation (RAG) chatbot for answering questions about Current Trends in Software Engineering (CTSE) lecture notes, featuring both Jupyter Notebook and Streamlit implementations.

## Features ✨

- **Dual Interface**: Interactive Jupyter Notebook & modern Streamlit web app
- **Advanced RAG**: Powered by Gemma:2b LLM and FAISS vector store
- **Lecture-Specific Answers**: Handles page-range queries (e.g., "Lecture 1, pages 2-4")
- **OCR Support**: Processes scanned PDFs using Tesseract
- **Persistent Storage**: Saves FAISS embeddings for faster reloads
- **Smart Chunking**: Semantic document splitting (300 chars, 50 overlap)
- **Strict Relevance**: Rejects non-CTSE questions with custom prompts

## Prerequisites 🛠️

- Python 3.8+
- Ollama (running Gemma:2b)
- Tesseract OCR (for scanned PDFs)
- 8GB+ RAM (16GB recommended)

## Installation ⚙️

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/CTSE_Chatbot.git
   cd CTSE_Chatbot
   ```

2. **Set up Ollama**:
   ```bash
   ollama pull gemma:2b
   ollama serve
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare lecture notes**:
   - Place PDFs in `./lecture_notes/`
   - For scanned PDFs, ensure Tesseract is installed:
     - **Windows**: [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
     - **Mac/Linux**: `brew install tesseract` or `sudo apt install tesseract-ocr`

## Usage 🚀

### Jupyter Notebook
```bash
jupyter notebook CTSE_Chatbot.ipynb
```
- Run cells sequentially
- Interactive Q&A via terminal-like interface

### Streamlit Web App
```bash
streamlit run app.py
```
- Open `http://localhost:8501` in browser
- Modern chat interface with message history

**Sample Questions**:
- "Explain zero padding in boundary handling"
- "Show content from Lecture 2, pages 5-7"
- "What are the 12 agile principles?"

## Project Structure 📂
```
CTSE_Chatbot/
├── lecture_notes/          # PDF lecture files
├── faiss_index/            # Saved vector embeddings
├── CTSE_Chatbot.ipynb      # Jupyter Notebook implementation
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Troubleshooting 🛠

| Issue | Solution |
|-------|----------|
| Ollama connection errors | Ensure `ollama serve` is running |
| Missing PDFs | Verify files are in `./lecture_notes/` |
| OCR failures | Check Tesseract installation path |
| Memory issues | Reduce `chunk_size` in notebook |

## License 📜
MIT License - See [LICENSE](LICENSE) for details.

---

Developed for **SE4010: Current Trends in Software Engineering**  
Semester 2, 2025 | Fernando W.T.R.P - IT21278280

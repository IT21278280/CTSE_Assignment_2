
```markdown
# CTSE Lecture Notes Chatbot

## Overview

This project was developed for the **SE4010: Current Trends in Software Engineering** course (Semester 1, 2025) as part of **Assignment 2: AI/ML**. It features an enhanced **Retrieval-Augmented Generation (RAG)** chatbot that intelligently answers questions based on CTSE lecture notes.

Built with **LangChain**, **Ollama (Mistral model)**, and **FAISS**, the chatbot runs inside a **Jupyter Notebook** using **Visual Studio Code**.

---

## Features

- Interactive Interface: Input loop allows dynamic Q&A with an `exit` command.
- Persistent Vector Storage: FAISS embeddings saved to disk for quicker reuse.
- Semantic Chunking: Splits lecture notes into meaningful chunks for accurate context.
- Advanced Embeddings: Uses `sentence-transformers/all-mpnet-base-v2` for top-tier semantic search.
- Custom Prompting: Focuses on concise, lecture-relevant responses.
- Error Handling: Detects missing files, invalid PDFs, or Ollama startup issues.
- Progress Bars & Logging: Visual feedback via `tqdm` and logging for traceability.

---

## Prerequisites

### Hardware
- Minimum 8GB RAM (16GB recommended)
- 10–20GB free disk space for models and FAISS index

### Software
- Python 3.8+
- Visual Studio Code with Python & Jupyter extensions
- [Ollama](https://ollama.com/)
- Git (optional for version control)

---

## Setup Instructions

### Clone the Repository
```bash
git clone <repository-url>
cd CTSE_Chatbot_Project
```

### Install Python
- Download: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Verify:
```bash
python --version  # Should be 3.8 or higher
```
- In VS Code: Press `Ctrl+Shift+P` > Python: Select Interpreter

### Install Ollama
```bash
# Download and install Ollama from https://ollama.com/
ollama pull mistral
ollama serve
```

### Install Dependencies
```bash
pip install langchain langchain-community langchain-ollama faiss-cpu pypdf sentence-transformers tqdm
```

### Prepare Lecture Notes
Create a folder and add PDFs:
```
CTSE_Chatbot_Project/
├── lecture_notes/
│   ├── lecture1.pdf
│   ├── lecture2.pdf
│   └── ...
├── faiss_index/      # Created automatically
└── CTSE_Chatbot.ipynb
```

---

## Usage

### Open the Notebook
- Launch Visual Studio Code
- Open `CTSE_Chatbot.ipynb`
- Select the correct Python kernel (top-right)

### Run the Notebook
- Ensure `ollama serve` is running
- Run each cell in order:

| Cell | Function |
|------|----------|
| Cell 3 | Import libraries & set up logging |
| Cell 5 | Load & chunk PDFs |
| Cell 7 | Create/load FAISS index |
| Cell 9 | Initialize Mistral LLM |
| Cell 11 | Build the RAG pipeline |
| Cell 13 | Start the interactive chatbot loop |
| Cell 15 | Test predefined questions |

---

## Interact with the Chatbot

Run **Cell 13** and enter questions such as:
```text
What is DevOps?
Summarize Lecture 1.
```
Type `exit` to quit.

Example:
```
CTSE Chatbot: Ask a question about the lecture notes (type 'exit' to quit)
Question: What is the main topic of Lecture 1?
Answer: [Answer based on lecture notes]
Source: [First 200 characters of source chunk]...
```

---

## Test Predefined Questions
Run **Cell 15** to automatically test sample questions with a progress bar.

---

## Enhancements

- Persistent FAISS index for faster startup  
- Semantic chunking with intelligent delimiters (`\n\n`, `\n`, `.`)  
- Improved accuracy using `all-mpnet-base-v2` embeddings  
- Custom prompts ensure relevance to lecture content  
- Comprehensive error handling and logging  
- Continuous interaction within the notebook  
- `tqdm` progress bars for enhanced UX  

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama Not Running | Run `ollama serve` and confirm with `ollama list` |
| PDF Loading Error | Ensure `lecture_notes/` exists and contains valid, text-based PDFs |
| Memory Issues | Pull a smaller model: `ollama pull gemma:2b` and use:<br>`llm = OllamaLLM(model="gemma:2b", temperature=0.3)`<br>Or reduce `chunk_size` to `300` in Cell 5 |
| FAISS Index Error | Delete the `./faiss_index` folder and rerun Cell 7 |
| Interactive Loop Not Prompting | Restart the kernel and rerun all cells |

---

## Deliverables for SE4010 Assignment

- `CTSE_Chatbot.ipynb` — main implementation notebook  
- `CTSE_Report.pdf` — report explaining design decisions  
- `CTSE_Demo.mp4` — short (2–3 min) demo video  

---

## Acknowledgments

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Ollama Documentation](https://ollama.com/)
- [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers)

---

### Developed by *[Your Name]* for SE4010, Semester 1, 2025.
```























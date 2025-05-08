# CTSE Lecture Notes Chatbot

## Overview

This project is developed for the **SE4010 - Current Trends in Software Engineering** course (Semester 1, 2025) as part of **Assignment 2: AI/ML**. It implements an enhanced **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on CTSE lecture notes. The chatbot is built using **LangChain**, **Ollama (Mistral model)**, and **FAISS**, and runs as a **Jupyter Notebook** in **Visual Studio Code (VS Code)**.

---

## Features

- **Interactive Interface**: Ask questions via an input loop, with `exit` to quit.
- **Persistent Vector Storage**: Saves FAISS embeddings to disk for faster startup.
- **Semantic Chunking**: Splits lecture notes into meaningful chunks for better context.
- **Advanced Embeddings**: Uses `sentence-transformers/all-mpnet-base-v2` for improved retrieval.
- **Custom Prompt**: Ensures concise, lecture-note-based answers.
- **Error Handling**: Robust checks for missing files, PDFs, or Ollama server issues.
- **Progress Bars & Logging**: Visual feedback and debugging logs for transparency.

---

## Prerequisites

### Hardware
- 8GB RAM minimum (16GB recommended).
- 10-20GB free disk space for Ollama models and FAISS index.

### Software
- **Python 3.8 or higher**: Download from [https://www.python.org/downloads/](https://www.python.org/downloads/).
- **Visual Studio Code**: Install from [https://code.visualstudio.com/](https://code.visualstudio.com/) with Python and Jupyter extensions.
- **Ollama**: Download from [https://ollama.com/](https://ollama.com/).
- **Git**: Optional, for version control ([https://git-scm.com/](https://git-scm.com/)).

---

## Setup Instructions

1. **Clone the Repository (if using Git)**:
   ```bash
   git clone <repository-url>
   cd CTSE_Chatbot_Project


Install Python:

Download from https://www.python.org/downloads/.
Verify:python --version

Should show 3.8 or higher.
In VS Code, select the Python interpreter (Ctrl+Shift+P > Python: Select Interpreter).


Install Ollama:

Download and install from https://ollama.com/.
Pull the Mistral model:ollama pull mistral


Start the Ollama server:ollama serve




Install Dependencies:

Open a terminal in VS Code (Terminal > New Terminal).
Install required libraries:pip install langchain langchain-community langchain-ollama faiss-cpu pypdf sentence-transformers tqdm




Prepare Lecture Notes:

Create a lecture_notes/ folder in the project root.
Place text-based PDF lecture notes (e.g., lecture1.pdf) in this folder.
Example structure:CTSE_Chatbot_Project/
├── lecture_notes/
│   ├── lecture1.pdf
│   ├── lecture2.pdf
│   └── ...
├── faiss_index/  (created after first run)
└── CTSE_Chatbot.ipynb






Usage

Open the Notebook:

Launch VS Code and open CTSE_Chatbot.ipynb.
Ensure the Python interpreter is set (top-right kernel selector).


Run the Notebook:

Ensure ollama serve is running in a terminal.
Run each cell sequentially (Ctrl+Enter or “Run Cell” button).
Key cells:
Cell 3: Imports libraries and sets up logging.
Cell 5: Loads and chunks PDFs.
Cell 7: Creates or loads the FAISS index.
Cell 9: Initializes the Mistral LLM.
Cell 11: Builds the RAG pipeline with a custom prompt.
Cell 13: Interactive loop for asking questions.
Cell 15: Tests predefined questions with a progress bar.




Interact with the Chatbot:

Run Cell 13 to start the interactive loop.
Enter questions like:What is DevOps?
Summarize Lecture 1.


Type exit to quit.
Example output:CTSE Chatbot: Ask a question about the lecture notes (type 'exit' to quit)
Question: What is the main topic of Lecture 1?
Answer: [Answer based on lecture notes]
Source: [First 200 characters of source chunk]...




Test Predefined Questions:

Run Cell 15 to see answers for example questions with a progress bar.




Enhancements
This chatbot improves on the basic version with:

Persistent FAISS Index: Saves embeddings to ./faiss_index for faster reloads.
Semantic Chunking: Uses smaller chunks (500 chars) with semantic separators (\n\n, \n, .) for better context.
Advanced Embeddings: all-mpnet-base-v2 improves retrieval accuracy over all-MiniLM-L6-v2.
Custom Prompt: Ensures answers are concise and lecture-note-specific.
Error Handling: Catches missing folders, empty PDFs, and Ollama errors.
Interactive Loop: Allows continuous questioning in the notebook.
Progress Bars: tqdm provides visual feedback for loading and testing.
Logging: Detailed logs for debugging and transparency.


Troubleshooting

Ollama Not Running:

Error: Connection refused.
Fix: Run ollama serve in a terminal and verify with:ollama list




PDF Loading Error:

Error: No documents loaded.
Fix: Ensure lecture_notes/ exists with text-based PDFs.


Memory Issues:

Fix: Use a smaller model (gemma:2b):ollama pull gemma:2b

Update Cell 9:llm = OllamaLLM(model="gemma:2b", temperature=0.3)

Or reduce chunk_size to 300 in Cell 5.


FAISS Index Error:

Fix: Delete ./faiss_index and rerun Cell 7.


Interactive Loop Not Prompting:

Fix: Restart the kernel (top-right “Restart” button) and rerun cells.




Deliverables for SE4010 Assignment

Jupyter Notebook: CTSE_Chatbot.ipynb (this project).
Report: CTSE_Report.pdf (justifies choices and enhancements).
Video Demo: CTSE_Demo.mp4 (2-3 minute demo in VS Code).


Acknowledgments

LangChain RAG Tutorial: https://python.langchain.com/docs/tutorials/rag/
Ollama Documentation: https://ollama.com/
Hugging Face Sentence Transformers: https://huggingface.co/sentence-transformers

Developed by [Your Name] for SE4010, Semester 1, 2025.







<!-- CTSE Lecture Notes Chatbot
Overview
This project is developed for the SE4010 Current Trends in Software Engineering course (Semester 1, 2025) as part of Assignment 2: AI/ML. It implements an enhanced Retrieval-Augmented Generation (RAG) chatbot that answers questions based on CTSE lecture notes. The chatbot is built using LangChain, Ollama (Mistral model), and FAISS, and runs as a Jupyter Notebook in Visual Studio Code (VS Code).
Features

Interactive Interface: Ask questions via an input loop, with exit to quit.
Persistent Vector Storage: Saves FAISS embeddings to disk for faster startup.
Semantic Chunking: Splits lecture notes into meaningful chunks for better context.
Advanced Embeddings: Uses sentence-transformers/all-mpnet-base-v2 for improved retrieval.
Custom Prompt: Ensures concise, lecture-note-based answers.
Error Handling: Robust checks for missing files, PDFs, or Ollama server issues.
Progress Bars & Logging: Visual feedback and debugging logs for transparency.

Prerequisites

Hardware:
8GB RAM minimum (16GB recommended).
10-20GB free disk space for Ollama models and FAISS index.


Software:
Python 3.8 or higher.
Visual Studio Code with Python and Jupyter extensions.
Ollama (https://ollama.com/).
Git (optional, for version control).



Setup Instructions

Clone the Repository (if using Git):
git clone <repository-url>
cd CTSE_Chatbot_Project


Install Python:

Download from https://www.python.org/downloads/.
Verify: python --version (should show 3.8+).
In VS Code, select the Python interpreter (Ctrl+Shift+P > Python: Select Interpreter).


Install Ollama:

Download and install from https://ollama.com/.
Pull the Mistral model:ollama pull mistral


Start the Ollama server:ollama serve




Install Dependencies:

Open a terminal in VS Code (Terminal > New Terminal).
Install required libraries:pip install langchain langchain-community langchain-ollama faiss-cpu pypdf sentence-transformers tqdm




Prepare Lecture Notes:

Create a lecture_notes/ folder in the project root.
Place text-based PDF lecture notes (e.g., lecture1.pdf) in this folder.
Example structure:CTSE_Chatbot_Project/
├── lecture_notes/
│   ├── lecture1.pdf
│   ├── lecture2.pdf
│   └── ...
├── faiss_index/  (created after first run)
└── CTSE_Chatbot.ipynb





Usage

Open the Notebook:

Launch VS Code and open CTSE_Chatbot.ipynb.
Ensure the Python interpreter is set (top-right kernel selector).


Run the Notebook:

Ensure ollama serve is running in a terminal.
Run each cell sequentially (Ctrl+Enter or “Run Cell” button).
Key cells:
Cell 3: Imports libraries and sets up logging.
Cell 5: Loads and chunks PDFs.
Cell 7: Creates or loads the FAISS index.
Cell 9: Initializes the Mistral LLM.
Cell 11: Builds the RAG pipeline with a custom prompt.
Cell 13: Interactive loop for asking questions.
Cell 15: Tests predefined questions with a progress bar.




Interact with the Chatbot:

Run Cell 13 to start the interactive loop.
Enter questions like “What is DevOps?” or “Summarize Lecture 1.”
Type exit to quit.
Example output:CTSE Chatbot: Ask a question about the lecture notes (type 'exit' to quit)
Question: What is the main topic of Lecture 1?
Answer: [Answer based on lecture notes]
Source: [First 200 characters of source chunk]...




Test Predefined Questions:

Run Cell 15 to see answers for example questions with a progress bar.



Enhancements
This chatbot improves on the basic version with:

Persistent FAISS Index: Saves embeddings to ./faiss_index for faster reloads.
Semantic Chunking: Uses smaller chunks (500 chars) with semantic separators (\n\n, \n, .) for better context.
Advanced Embeddings: all-mpnet-base-v2 improves retrieval accuracy over all-MiniLM-L6-v2.
Custom Prompt: Ensures answers are concise and lecture-note-specific.
Error Handling: Catches missing folders, empty PDFs, and Ollama errors.
Interactive Loop: Allows continuous questioning in the notebook.
Progress Bars: tqdm provides visual feedback for loading and testing.
Logging: Detailed logs for debugging and transparency.

Troubleshooting

Ollama Not Running:
Error: Connection refused.
Fix: Run ollama serve in a terminal and verify with ollama list.


PDF Loading Error:
Error: No documents loaded.
Fix: Ensure lecture_notes/ exists with text-based PDFs.


Memory Issues:
Fix: Use a smaller model (gemma:2b):ollama pull gemma:2b

Update Cell 9:llm = OllamaLLM(model="gemma:2b", temperature=0.3)


Or reduce chunk_size to 300 in Cell 5.


FAISS Index Error:
Fix: Delete ./faiss_index and rerun Cell 7.


Interactive Loop Not Prompting:
Fix: Restart the kernel (top-right “Restart” button) and rerun cells.



Deliverables for SE4010 Assignment

Jupyter Notebook: CTSE_Chatbot.ipynb (this project).
Report: CTSE_Report.pdf (justifies choices and enhancements).
Video Demo: CTSE_Demo.mp4 (2-3 minute demo in VS Code).

Acknowledgments

LangChain RAG Tutorial: https://python.langchain.com/docs/tutorials/rag/
Ollama Documentation: https://ollama.com/
Hugging Face Sentence Transformers: https://huggingface.co/sentence-transformers


Developed by [Your Name] for SE4010, Semester 1, 2025. -->

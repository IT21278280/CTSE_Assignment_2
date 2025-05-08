CTSE Lecture Notes Chatbot
Overview
This project implements an enhanced Retrieval-Augmented Generation (RAG) chatbot for the SE4010 AI/ML Assignment (Semester 1, 2025). The chatbot answers questions based on Current Trends in Software Engineering (CTSE) lecture notes, using LangChain, Ollama (Mistral model), and FAISS for vector storage. It is developed as a Jupyter Notebook in Visual Studio Code, optimized for functionality, performance, and user experience.
Features

Semantic Chunking: Splits lecture notes into meaningful chunks (500 characters) for better context retrieval.
Persistent Vector Storage: Saves FAISS index to disk, avoiding recomputation.
Advanced Embeddings: Uses sentence-transformers/all-mpnet-base-v2 for improved semantic understanding.
Custom Prompt: Ensures concise, accurate answers based solely on lecture notes.
Interactive Interface: Allows continuous question input with an exit option.
Error Handling: Robust checks for missing PDFs, Ollama server, and processing errors.
Progress Bars & Logging: Visual feedback with tqdm and detailed logs for debugging.
Optimized Retrieval: Retrieves top 3 relevant chunks for balanced speed and accuracy.

Prerequisites

Software:
Python 3.8 or higher
Visual Studio Code with Python and Jupyter extensions
Ollama (https://ollama.com/)


Hardware:
Minimum 8GB RAM (16GB recommended)
10-20GB free disk space for models and indexes


Libraries:pip install langchain langchain-community langchain-ollama faiss-cpu pypdf sentence-transformers tqdm


Lecture Notes:
Text-based PDF files in the lecture_notes/ folder



Setup Instructions

Clone the Repository (if using Git):git clone <repository-url>
cd CTSE_Chatbot_Project


Install Python:
Download from https://www.python.org/downloads/.
Verify: python --version


Install Ollama:
Download from https://ollama.com/.
Pull the Mistral model:ollama pull mistral


Start the Ollama server:ollama serve




Install Libraries:pip install langchain langchain-community langchain-ollama faiss-cpu pypdf sentence-transformers tqdm


Prepare Lecture Notes:
Create a lecture_notes/ folder in the project root.
Add text-based PDFs (e.g., lecture1.pdf).


Set Up VS Code:
Open VS Code and install Python and Jupyter extensions.
Open CTSE_Chatbot.ipynb and select the Python interpreter (Ctrl+Shift+P > Python: Select Interpreter).



Project Structure
CTSE_Chatbot_Project/
├── lecture_notes/          # Folder for lecture note PDFs
├── faiss_index/           # Persistent FAISS index (created after first run)
├── CTSE_Chatbot.ipynb     # Main Jupyter Notebook
└── README.md              # Project documentation

Usage

Open the Notebook:
In VS Code, open CTSE_Chatbot.ipynb.
Ensure ollama serve is running in a terminal.


Run Cells:
Execute cells sequentially (Ctrl+Enter or "Run Cell").
Key cells:
Cell 3: Imports libraries.
Cell 5: Loads and chunks PDFs.
Cell 7: Sets up embeddings and FAISS index.
Cell 9: Initializes Ollama LLM.
Cell 11: Builds RAG pipeline.
Cell 13: Interactive loop for asking questions.
Cell 15: Tests predefined questions.




Interact with the Chatbot:
Run Cell 13 to enter questions (e.g., "What is DevOps?").
Type exit to quit.
Cell 15 automatically tests sample questions with a progress bar.


View Outputs:
Answers and source document snippets appear below cells.
Logs provide debugging information.



Troubleshooting

Ollama Not Running:
Error: Connection refused.
Fix: Run ollama serve in a terminal.


PDF Loading Error:
Error: No documents loaded.
Fix: Ensure lecture_notes/ contains text-based PDFs.


Memory Issues:
Fix: Use gemma:2b model (ollama pull gemma:2b, update Cell 9 to llm = OllamaLLM(model="gemma:2b", temperature=0.3)).
Fix: Reduce chunk_size to 300 in Cell 5.


FAISS Index Error:
Fix: Delete faiss_index/ and rerun Cell 7.


VS Code Input Issues:
Fix: Restart the Jupyter kernel (top-right "Restart" button).



Assignment Details
This project fulfills the SE4010 AI/ML Assignment 2 requirements:

Deliverables:
CTSE_Chatbot.ipynb: Enhanced chatbot implementation.
CTSE_Report.pdf: Justification report (update to reflect enhancements).
CTSE_Demo.mp4: 2-3 minute video showing the chatbot in VS Code.


Marking Scheme:
Working Notebook: 40%
LLM Choice Justification: 20%
Development Approach: 20%
GenAI Transparency: 10%
Video Demonstration: 10%



License
MIT License. See LICENSE for details (if included).

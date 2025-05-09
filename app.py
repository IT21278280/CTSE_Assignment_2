import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Title of the app with custom CSS
st.markdown(
    """
    <style>
    .title {
        color: #4CAF50;
        font-size: 36px;
        text-align: center;
    }
    .chat-container {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e8f0fe;
        padding: 5px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .bot-message {
        background-color: #d4edda;
        padding: 5px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    <h1 class='title'>CTSE Lecture Notes Chatbot (LLaMA3 + LangChain)</h1>
    """,
    unsafe_allow_html=True
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history with styling
for chat in st.session_state.chat_history:
    st.markdown(f"<div class='chat-container'><div class='user-message'>**You:** {chat['question']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-message'>**Bot:** {chat['answer']}</div>", unsafe_allow_html=True)
    if 'source' in chat:
        st.markdown(f"<div class='bot-message'>**Source:** {chat['source']}</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

# Load and process lecture notes (run once)
@st.cache_data
def load_and_process_documents():
    logging.info("Loading and processing lecture notes...")
    notes_dir = './lecture_notes/'
    if not os.path.exists(notes_dir):
        logging.error(f"Directory {notes_dir} not found.")
        st.error(f"Directory {notes_dir} not found. Please create it and add PDF files.")
        return None
    loader = PyPDFDirectoryLoader(notes_dir)
    try:
        documents = loader.load()
        if not documents:
            logging.error("No documents loaded.")
            st.error("No documents loaded. Ensure PDFs are text-based and not empty.")
            return None
    except Exception as e:
        logging.error(f"Error loading PDFs: {e}")
        st.error(f"Error loading PDFs: {e}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

@st.cache_resource
def setup_vector_store(_chunks):  # Use _chunks to avoid hashing
    logging.info("Setting up vector store...")
    faiss_index_path = './faiss_index'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if os.path.exists(faiss_index_path):
        vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_documents(_chunks, embeddings)
        vector_store.save_local(faiss_index_path)
    return vector_store

@st.cache_resource
def setup_llm():
    logging.info("Initializing Ollama LLM...")
    try:
        return OllamaLLM(model="mistral", temperature=0.3)
    except Exception as e:
        logging.error(f"Error initializing Ollama: {e}")
        st.error(f"Error initializing Ollama: {e}")
        return None

@st.cache_resource
def setup_qa_chain(vector_store, llm):
    logging.info("Building RAG pipeline...")
    prompt_template = """
    You are a knowledgeable assistant for Current Trends in Software Engineering (CTSE). Answer the following question based solely on the provided lecture notes. Provide a clear, concise, and accurate response. If the information is not available, say so.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# Load data on app start
chunks = load_and_process_documents()
if chunks:
    vector_store = setup_vector_store(chunks)
    llm = setup_llm()
    if vector_store and llm:
        qa_chain = setup_qa_chain(vector_store, llm)
    else:
        st.error("Failed to set up chatbot. Check logs.")
else:
    st.stop()

# Function to ask questions
def ask_question(question):
    if qa_chain:
        try:
            result = qa_chain({"query": question})
            answer = result["result"].strip()
            source = result["source_documents"][0].page_content[:200] + "..." if result["source_documents"] else "No source available."
            logging.info(f"Question: {question}")
            logging.info(f"Answer: {answer}")
            return answer, source
        except Exception as e:
            logging.error(f"Error processing question: {e}")
            return f"Error: Unable to process question. {e}", ""
    return "Chatbot not initialized.", ""

# Clear chat history button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Input for user question
question = st.text_input("Ask a question about CTSE Software Engineering:")
if st.button("Ask"):
    if question:
        answer, source = ask_question(question)
        st.session_state.chat_history.append({"question": question, "answer": answer, "source": source})
        st.experimental_rerun()

# import streamlit as st
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from tqdm import tqdm
# import os
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Title of the app
# st.title("CTSE Lecture Notes Chatbot (LLaMA3 + LangChain)")

# # Initialize session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display chat history
# for chat in st.session_state.chat_history:
#     st.write(f"**You:** {chat['question']}")
#     st.write(f"**Bot:** {chat['answer']}")
#     if 'source' in chat:
#         st.write(f"**Source:** {chat['source']}")
#     st.write("---")

# # Load and process lecture notes (run once)
# @st.cache_data
# def load_and_process_documents():
#     logging.info("Loading and processing lecture notes...")
#     notes_dir = './lecture_notes/'
#     if not os.path.exists(notes_dir):
#         logging.error(f"Directory {notes_dir} not found.")
#         st.error(f"Directory {notes_dir} not found. Please create it and add PDF files.")
#         return None
#     loader = PyPDFDirectoryLoader(notes_dir)
#     try:
#         documents = loader.load()
#         if not documents:
#             logging.error("No documents loaded.")
#             st.error("No documents loaded. Ensure PDFs are text-based and not empty.")
#             return None
#     except Exception as e:
#         logging.error(f"Error loading PDFs: {e}")
#         st.error(f"Error loading PDFs: {e}")
#         return None

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])
#     chunks = text_splitter.split_documents(documents)
#     logging.info(f"Created {len(chunks)} chunks.")
#     return chunks

# @st.cache_resource
# def setup_vector_store(chunks):
#     logging.info("Setting up vector store...")
#     faiss_index_path = './faiss_index'
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     if os.path.exists(faiss_index_path):
#         vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
#     else:
#         vector_store = FAISS.from_documents(chunks, embeddings)
#         vector_store.save_local(faiss_index_path)
#     return vector_store

# @st.cache_resource
# def setup_llm():
#     logging.info("Initializing Ollama LLM...")
#     try:
#         return OllamaLLM(model="mistral", temperature=0.3)
#     except Exception as e:
#         logging.error(f"Error initializing Ollama: {e}")
#         st.error(f"Error initializing Ollama: {e}")
#         return None

# @st.cache_resource
# def setup_qa_chain(vector_store, llm):
#     logging.info("Building RAG pipeline...")
#     prompt_template = """
#     You are a knowledgeable assistant for Current Trends in Software Engineering (CTSE). Answer the following question based solely on the provided lecture notes. Provide a clear, concise, and accurate response. If the information is not available, say so.

#     Context: {context}
#     Question: {question}

#     Answer:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt}
#     )

# # Load data on app start
# chunks = load_and_process_documents()
# if chunks:
#     vector_store = setup_vector_store(chunks)
#     llm = setup_llm()
#     if vector_store and llm:
#         qa_chain = setup_qa_chain(vector_store, llm)
#     else:
#         st.error("Failed to set up chatbot. Check logs.")
# else:
#     st.stop()

# # Function to ask questions
# def ask_question(question):
#     if qa_chain:
#         try:
#             result = qa_chain({"query": question})
#             answer = result["result"].strip()
#             source = result["source_documents"][0].page_content[:200] + "..." if result["source_documents"] else "No source available."
#             logging.info(f"Question: {question}")
#             logging.info(f"Answer: {answer}")
#             return answer, source
#         except Exception as e:
#             logging.error(f"Error processing question: {e}")
#             return f"Error: Unable to process question. {e}", ""
#     return "Chatbot not initialized.", ""

# # Input for user question
# question = st.text_input("Ask a question about CTSE Software Engineering:")
# if st.button("Ask"):
#     if question:
#         answer, source = ask_question(question)
#         st.session_state.chat_history.append({"question": question, "answer": answer, "source": source})
#         st.experimental_rerun()
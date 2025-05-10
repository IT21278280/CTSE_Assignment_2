import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from pdf2image import convert_from_path
import pytesseract
import PyPDF2
import os
import logging
import re
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Specify Tesseract path (Windows example; adjust for your system)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment and adjust if needed

# Custom CSS for enhanced UI/UX
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .chat-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .source-box {
        background-color: #f9f9f9;
        padding: 10px;
        border-left: 4px solid #4CAF50;
        border-radius: 5px;
        font-size: 14px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with emoji and styling
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>CTSE Lecture Notes Chatbot ✨ (LLaMA3 + LangChain)</h1>", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history with enhanced UI/UX
with st.expander("Chat History", expanded=True):
    if not st.session_state.chat_history:
        st.markdown("<p style='color: #666;'>No questions asked yet. Start by asking a question below!</p>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        with st.container():
            st.markdown(
                f"""
                <div class="chat-container">
                    <strong style="color: #2c3e50;">You:</strong> {chat['question']}<br>
                    <strong style="color: #2c3e50;">Bot:</strong> {chat['answer']}<br>
                    <div class="source-box">
                        <strong>Source:</strong><br>
                        {chat['source']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Clean text function to remove artifacts and improve formatting
def clean_text(text):
    text = re.sub(r'\n+', '\n', text.strip())
    text = re.sub(r'●|\○|\-|\+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.capitalize()

# Load and process lecture notes with page tracking and OCR fallback
@st.cache_data
def load_and_process_documents():
    logging.info("Loading and processing lecture notes...")
    notes_dir = './lecture_notes/'
    if not os.path.exists(notes_dir):
        logging.error(f"Directory {notes_dir} not found.")
        st.error(f"Directory {notes_dir} not found. Please create it and add PDF files.")
        return None

    documents = []
    for pdf_file in os.listdir(notes_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(notes_dir, pdf_file)
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_extracted = False
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text and text.strip():
                            cleaned_text = clean_text(text)
                            doc = Document(page_content=cleaned_text, metadata={"source": pdf_file, "page": page_num + 1})
                            documents.append(doc)
                            text_extracted = True
                        else:
                            logging.warning(f"No text extracted from {pdf_file}, page {page_num + 1}. Falling back to OCR.")
                            break

                if not text_extracted:
                    logging.info(f"Using OCR for {pdf_file}...")
                    images = convert_from_path(pdf_path)
                    for page_num, image in enumerate(images):
                        text = pytesseract.image_to_string(image)
                        if text and text.strip():
                            cleaned_text = clean_text(text)
                            doc = Document(page_content=cleaned_text, metadata={"source": pdf_file, "page": page_num + 1})
                            documents.append(doc)
                        else:
                            logging.warning(f"OCR failed to extract text from {pdf_file}, page {page_num + 1}.")
            except Exception as e:
                logging.error(f"Error processing {pdf_file}: {e}")
                st.error(f"Error processing {pdf_file}: {e}")
                continue

    if not documents:
        logging.error("No documents loaded after processing.")
        st.error("No documents loaded. Ensure PDFs are text-based or OCR-compatible.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""])
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
    logging.info("Initializing Ollama LLM with Gemma:2b...")
    try:
        return OllamaLLM(model="gemma:2b", temperature=0.3)
    except Exception as e:
        logging.error(f"Error initializing Ollama: {e}")
        st.error(f"Error initializing Ollama: {e}")
        return None

@st.cache_resource
def setup_qa_chain(_vector_store, _llm):  # Use _vector_store and _llm to avoid hashing
    logging.info("Building RAG pipeline...")
    prompt_template = """
    You are a specialized assistant for Current Trends in Software Engineering (CTSE) lecture notes. Your task is to provide a clear, concise, and accurate answer using *only* the exact text from the provided CTSE lecture notes. Copy the relevant sentence(s) or phrase(s) directly from the context without adding, modifying, or elaborating on the content. Use bullet points or short sentences for clarity. If the question is unrelated to CTSE lecture notes (e.g., sports, general knowledge), respond exactly with: 'This question is outside the scope of CTSE lecture notes. Please ask about topics from the lecture notes.' If no relevant information is found, respond exactly with: 'No relevant information found in the CTSE lecture notes.'

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(search_kwargs={"k": 3}),
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

# Function to ask questions with improved retrieval and concise output
def ask_question(question):
    if qa_chain:
        try:
            # Extract lecture name and page range from the question (if applicable)
            lecture_match = re.search(r'Lecture(\d+)', question, re.IGNORECASE)
            page_range = None
            if "first 2 page" in question.lower() or "page 1 to 2" in question.lower():
                page_range = [1, 2]

            if lecture_match and page_range:
                lecture_num = lecture_match.group(1)
                lecture_file = f"lecture{lecture_num}.pdf"
                filtered_docs = [doc for doc in chunks if doc.metadata["source"] == lecture_file and doc.metadata["page"] in page_range]
                if not filtered_docs:
                    return "No content found for the first two pages of Lecture {}.".format(lecture_num), ""
                temp_vector_store = FAISS.from_documents(filtered_docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
                temp_qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=temp_vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": qa_chain.combine_documents_chain.llm_chain.prompt}
                )
                result = temp_qa_chain.invoke({"query": question})
            else:
                # For general questions, use the main vector store with a refined keyword filter
                keywords = [word for word in question.lower().split() if word not in ['what', 'is', 'in', 'the', 'a', 'an']]
                filtered_docs = [
                    doc for doc in chunks
                    if any(keyword in doc.page_content.lower() for keyword in keywords)
                ]
                if not filtered_docs:
                    return "No relevant information found in the CTSE lecture notes.", ""
                temp_vector_store = FAISS.from_documents(filtered_docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
                temp_qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=temp_vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": qa_chain.combine_documents_chain.llm_chain.prompt}
                )
                result = temp_qa_chain.invoke({"query": question})

            answer = result["result"].strip()
            answer = clean_text(answer)  # Ensure clean and concise output
            source = result["source_documents"][0].page_content[:200] + "..." if result["source_documents"] else "No source available."
            source = clean_text(source)
            logging.info(f"Question: {question}")
            logging.info(f"Answer: {answer}")
            return answer, source
        except Exception as e:
            logging.error(f"Error processing question: {e}")
            return f"Error: Unable to process question. {e}", ""
    return "Chatbot not initialized.", ""

# Input for user question with styled button
question = st.text_input("Ask a question about CTSE Software Engineering:", key="question_input")
if st.button("Ask", key="ask_button"):
    if question:
        answer, source = ask_question(question)
        st.session_state.chat_history.append({"question": question, "answer": answer, "source": source})
        st.rerun()











# import streamlit as st
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.docstore.document import Document
# from pdf2image import convert_from_path
# import pytesseract
# import PyPDF2
# import os
# import logging
# import re
# from tqdm import tqdm

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Specify Tesseract path (Windows example; adjust for your system)
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment and adjust if needed

# # Custom CSS for enhanced UI/UX
# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #f0f2f6;
#         padding: 20px;
#         border-radius: 10px;
#     }
#     .stTextInput > div > div > input {
#         border-radius: 5px;
#         padding: 10px;
#         font-size: 16px;
#     }
#     .stButton > button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-weight: bold;
#         font-size: 16px;
#     }
#     .stButton > button:hover {
#         background-color: #45a049;
#     }
#     .chat-container {
#         background-color: #ffffff;
#         padding: 15px;
#         border-radius: 8px;
#         margin-bottom: 10px;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#     }
#     .source-box {
#         background-color: #f9f9f9;
#         padding: 10px;
#         border-left: 4px solid #4CAF50;
#         border-radius: 5px;
#         font-size: 14px;
#         margin-top: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title with emoji and styling
# st.markdown("<h1 style='text-align: center; color: #2c3e50;'>CTSE Lecture Notes Chatbot ✨ (LLaMA3 + LangChain)</h1>", unsafe_allow_html=True)

# # Initialize session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display chat history with enhanced UI/UX
# with st.expander("Chat History", expanded=True):
#     if not st.session_state.chat_history:
#         st.markdown("<p style='color: #666;'>No questions asked yet. Start by asking a question below!</p>", unsafe_allow_html=True)
#     for chat in st.session_state.chat_history:
#         with st.container():
#             st.markdown(
#                 f"""
#                 <div class="chat-container">
#                     <strong style="color: #2c3e50;">You:</strong> {chat['question']}<br>
#                     <strong style="color: #2c3e50;">Bot:</strong> {chat['answer']}<br>
#                     <div class="source-box">
#                         <strong>Source:</strong><br>
#                         {chat['source']}
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )

# # Clean text function to remove artifacts and improve formatting
# def clean_text(text):
#     text = re.sub(r'\n+', '\n', text.strip())
#     text = re.sub(r'●|\○|\-|\+', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.capitalize()

# # Load and process lecture notes with page tracking and OCR fallback
# @st.cache_data
# def load_and_process_documents():
#     logging.info("Loading and processing lecture notes...")
#     notes_dir = './lecture_notes/'
#     if not os.path.exists(notes_dir):
#         logging.error(f"Directory {notes_dir} not found.")
#         st.error(f"Directory {notes_dir} not found. Please create it and add PDF files.")
#         return None

#     documents = []
#     for pdf_file in os.listdir(notes_dir):
#         if pdf_file.endswith('.pdf'):
#             pdf_path = os.path.join(notes_dir, pdf_file)
#             try:
#                 with open(pdf_path, 'rb') as file:
#                     pdf_reader = PyPDF2.PdfReader(file)
#                     text_extracted = False
#                     for page_num in range(len(pdf_reader.pages)):
#                         page = pdf_reader.pages[page_num]
#                         text = page.extract_text()
#                         if text and text.strip():
#                             cleaned_text = clean_text(text)
#                             doc = Document(page_content=cleaned_text, metadata={"source": pdf_file, "page": page_num + 1})
#                             documents.append(doc)
#                             text_extracted = True
#                         else:
#                             logging.warning(f"No text extracted from {pdf_file}, page {page_num + 1}. Falling back to OCR.")
#                             break

#                 if not text_extracted:
#                     logging.info(f"Using OCR for {pdf_file}...")
#                     images = convert_from_path(pdf_path)
#                     for page_num, image in enumerate(images):
#                         text = pytesseract.image_to_string(image)
#                         if text and text.strip():
#                             cleaned_text = clean_text(text)
#                             doc = Document(page_content=cleaned_text, metadata={"source": pdf_file, "page": page_num + 1})
#                             documents.append(doc)
#                         else:
#                             logging.warning(f"OCR failed to extract text from {pdf_file}, page {page_num + 1}.")
#             except Exception as e:
#                 logging.error(f"Error processing {pdf_file}: {e}")
#                 st.error(f"Error processing {pdf_file}: {e}")
#                 continue

#     if not documents:
#         logging.error("No documents loaded after processing.")
#         st.error("No documents loaded. Ensure PDFs are text-based or OCR-compatible.")
#         return None

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""])
#     chunks = text_splitter.split_documents(documents)
#     logging.info(f"Created {len(chunks)} chunks.")
#     return chunks

# @st.cache_resource
# def setup_vector_store(_chunks):  # Use _chunks to avoid hashing
#     logging.info("Setting up vector store...")
#     faiss_index_path = './faiss_index'
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     if os.path.exists(faiss_index_path):
#         vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
#     else:
#         vector_store = FAISS.from_documents(_chunks, embeddings)
#         vector_store.save_local(faiss_index_path)
#     return vector_store

# @st.cache_resource
# def setup_llm():
#     logging.info("Initializing Ollama LLM with TinyLLaMA...")
#     try:
#         return OllamaLLM(model="tinyllama", temperature=0.3)
#     except Exception as e:
#         logging.error(f"Error initializing Ollama: {e}")
#         st.error(f"Error initializing Ollama: {e}")
#         return None

# @st.cache_resource
# def setup_qa_chain(_vector_store, _llm):  # Use _vector_store and _llm to avoid hashing
#     logging.info("Building RAG pipeline...")
#     prompt_template = """
#     You are a knowledgeable assistant for Current Trends in Software Engineering (CTSE). Your task is to provide a clear, concise, and accurate answer to the user's question based solely on the provided lecture notes. Focus on the specific topic asked (e.g., horizontal scaling, load balancing) and avoid including unrelated topics like Hadoop components unless directly relevant. Use bullet points or short paragraphs for clarity, and keep the response concise. If the information is not available, state that clearly.

#     Context: {context}
#     Question: {question}

#     Answer:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return RetrievalQA.from_chain_type(
#         llm=_llm,
#         chain_type="stuff",
#         retriever=_vector_store.as_retriever(search_kwargs={"k": 2}),
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

# # Function to ask questions with improved retrieval and concise output
# def ask_question(question):
#     if qa_chain:
#         try:
#             # Extract lecture name and page range from the question (if applicable)
#             lecture_match = re.search(r'Lecture(\d+)', question, re.IGNORECASE)
#             page_range = None
#             if "first 2 page" in question.lower() or "page 1 to 2" in question.lower():
#                 page_range = [1, 2]

#             if lecture_match and page_range:
#                 lecture_num = lecture_match.group(1)
#                 lecture_file = f"lecture{lecture_num}.pdf"
#                 filtered_docs = [doc for doc in chunks if doc.metadata["source"] == lecture_file and doc.metadata["page"] in page_range]
#                 if not filtered_docs:
#                     return "No content found for the first two pages of Lecture {}.".format(lecture_num), ""
#                 temp_vector_store = FAISS.from_documents(filtered_docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
#                 temp_qa_chain = RetrievalQA.from_chain_type(
#                     llm=llm,
#                     chain_type="stuff",
#                     retriever=temp_vector_store.as_retriever(search_kwargs={"k": 2}),
#                     return_source_documents=True,
#                     chain_type_kwargs={"prompt": qa_chain.combine_documents_chain.llm_chain.prompt}
#                 )
#                 result = temp_qa_chain.invoke({"query": question})
#             else:
#                 # For general questions, use the main vector store with a keyword filter
#                 keywords = question.lower().split()
#                 filtered_docs = [
#                     doc for doc in chunks
#                     if any(keyword in doc.page_content.lower() for keyword in keywords)
#                 ]
#                 if not filtered_docs:
#                     return "No relevant information found in the lecture notes for this question.", ""
#                 temp_vector_store = FAISS.from_documents(filtered_docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
#                 temp_qa_chain = RetrievalQA.from_chain_type(
#                     llm=llm,
#                     chain_type="stuff",
#                     retriever=temp_vector_store.as_retriever(search_kwargs={"k": 2}),
#                     return_source_documents=True,
#                     chain_type_kwargs={"prompt": qa_chain.combine_documents_chain.llm_chain.prompt}
#                 )
#                 result = temp_qa_chain.invoke({"query": question})

#             answer = result["result"].strip()
#             answer = clean_text(answer)  # Ensure clean and concise output
#             source = result["source_documents"][0].page_content[:200] + "..." if result["source_documents"] else "No source available."
#             source = clean_text(source)
#             logging.info(f"Question: {question}")
#             logging.info(f"Answer: {answer}")
#             return answer, source
#         except Exception as e:
#             logging.error(f"Error processing question: {e}")
#             return f"Error: Unable to process question. {e}", ""
#     return "Chatbot not initialized.", ""

# # Input for user question with styled button
# question = st.text_input("Ask a question about CTSE Software Engineering:", key="question_input")
# if st.button("Ask", key="ask_button"):
#     if question:
#         answer, source = ask_question(question)
#         st.session_state.chat_history.append({"question": question, "answer": answer, "source": source})
#         st.rerun()





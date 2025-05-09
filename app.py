
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

# Title of the app with styling
st.title("CTSE Lecture Notes Chatbot :sparkles: (LLaMA3 + LangChain)")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history with improved formatting
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['question']}")
    st.write(f"**Bot:**", chat['answer'])
    if 'source' in chat:
        st.write("**Source:**")
        st.code(chat['source'], language="text")
    st.write("---")

# Clean text function to remove artifacts and improve formatting
def clean_text(text):
    # Remove excessive newlines and trailing whitespace
    text = re.sub(r'\n+', '\n', text.strip())
    # Remove bullet point symbols and normalize spacing
    text = re.sub(r'●|\○|\-|\+', '', text)
    text = re.sub(r'\s+', ' ', text)
    # Ensure proper sentence structure
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
                # First attempt: Extract text directly with PyPDF2
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

                # Fallback to OCR if direct extraction fails
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
    logging.info("Initializing Ollama LLM with TinyLLaMA...")
    try:
        return OllamaLLM(model="tinyllama", temperature=0.3)
    except Exception as e:
        logging.error(f"Error initializing Ollama: {e}")
        st.error(f"Error initializing Ollama: {e}")
        return None

@st.cache_resource
def setup_qa_chain(_vector_store, _llm):  # Use _vector_store and _llm to avoid hashing
    logging.info("Building RAG pipeline...")
    prompt_template = """
    You are a knowledgeable assistant for Current Trends in Software Engineering (CTSE). Answer the following question based solely on the provided lecture notes. Provide a clear, concise, and well-structured response with proper paragraphs and bullet points where applicable. If the information is not available, say so. If the question asks for content from specific pages, use the page metadata to filter the context and present the content in an organized manner.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(search_kwargs={"k": 2}),
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

# Function to ask questions with page filtering and beautified output
def ask_question(question):
    if qa_chain:
        try:
            # Check if the question asks for specific pages (e.g., "first 2 pages")
            page_range = None
            if "first 2 page" in question.lower() or "page 1 to 2" in question.lower():
                page_range = [1, 2]

            # Retrieve and filter documents by page if specified
            if page_range:
                filtered_docs = [doc for doc in chunks if doc.metadata["page"] in page_range]
                if not filtered_docs:
                    return "No content found for the specified pages in the lecture notes.", ""
                # Create a temporary vector store with filtered docs
                temp_vector_store = FAISS.from_documents(filtered_docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
                temp_qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=temp_vector_store.as_retriever(search_kwargs={"k": 2}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": qa_chain.combine_documents_chain.llm_chain.prompt}
                )
                result = temp_qa_chain.invoke({"query": question})
            else:
                result = qa_chain.invoke({"query": question})

            answer = result["result"].strip()
            # Clean the answer for better presentation
            answer = clean_text(answer)
            source = result["source_documents"][0].page_content[:200] + "..." if result["source_documents"] else "No source available."
            source = clean_text(source)
            logging.info(f"Question: {question}")
            logging.info(f"Answer: {answer}")
            return answer, source
        except Exception as e:
            logging.error(f"Error processing question: {e}")
            return f"Error: Unable to process question. {e}", ""
    return "Chatbot not initialized.", ""

# Input for user question
question = st.text_input("Ask a question about CTSE Software Engineering:")
if st.button("Ask"):
    if question:
        answer, source = ask_question(question)
        st.session_state.chat_history.append({"question": question, "answer": answer, "source": source})
        st.rerun()



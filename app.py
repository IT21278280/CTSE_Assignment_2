import streamlit as st

# Title of the app
st.title("CTSE Lecture Notes Chatbot (LLaMA3 + LangChain)")

# Chat history
st.session_state.chat_history = st.session_state.get("chat_history", [])

# Display chat history
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['question']}")
    st.write(f"**Bot:** {chat['answer']}")
    if 'source' in chat:
        st.write(f"**Source:** {chat['source']}")
    st.write("---")

# Input for user question
question = st.text_input("Ask a question about CTSE Software Engineering:")
if st.button("Ask"):
    if question:
        st.session_state.chat_history.append({"question": question})
        st.experimental_rerun()
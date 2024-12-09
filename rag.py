import os
import openai
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st

# Set your OpenAI API Key
openai.api_key = "your-openai-api-key"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to create FAISS index
def create_faiss_index(texts):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    return vector_store

# Function to load or create FAISS index
def get_faiss_index(pdf_files):
    texts = []
    for pdf_file in pdf_files:
        texts.append(extract_text_from_pdf(pdf_file))
    vector_store = create_faiss_index(texts)
    return vector_store

# Streamlit UI
st.title("RAG Chatbot with GPT-3.5 Turbo")
st.write("Upload PDFs and ask context-based questions.")

# File upload section
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

# Initialize FAISS index
if uploaded_files:
    st.write("Processing PDFs...")
    vector_store = get_faiss_index(uploaded_files)
    st.success("PDFs processed and indexed successfully!")

    # Query input
    user_query = st.text_input("Ask your question:")
    
    if user_query:
        # Create RetrievalQA pipeline
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            retriever=retriever,
            return_source_documents=True
        )

        # Generate answer
        result = qa_chain({"query": user_query})
        st.subheader("Answer:")
        st.write(result["result"])

        # Display retrieved documents (optional)
        with st.expander("Retrieved Context"):
            for doc in result["source_documents"]:
                st.write(doc.page_content)


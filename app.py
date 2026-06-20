import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()


os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


llm = ChatGroq(model_name="llama-3.1-8b-instant")


prompt= ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only
    please provide most accurate response based on the context
    <context>
    {context}
    <context>
    "Question": {input}
    
    """
)

with st.sidebar:
    upload_file = st.sidebar.file_uploader("📂 Upload Your File", type="pdf")
    if upload_file is not None:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
            temp_file.write(upload_file.getvalue())
            temp_path = temp_file.name


def create_vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFLoader(file_path=temp_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.document = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vector = FAISS.from_documents(st.session_state.document,st.session_state.embedding)
        
        

st.title("Next-Gen Document Intelligence: RAG Document Querying with Groq Acceleration 🗂️⚡🤖")

user_input=st.text_input("Ask a question about your document")



if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database Is Ready")
    
    
import time

if user_input:
    document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
    retriver=st.session_state.vector.as_retriever()
    chain = create_retrieval_chain(retriver,document_chain)
    
    start = time.process_time()
    response = chain.invoke({"input":user_input})
    print(f"Response_time {time.process_time()}-start")
    
    st.write(response['answer'])
    
    with st.expander("Document Similarity Search"):
        
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-------------------------")


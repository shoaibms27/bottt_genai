import streamlit as st
from langchain.document_loaders import PyPDFLoader  # Corrected import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from together import TogetherAPI  # Use TogetherAPI class if applicable

# Streamlit app title
st.title("🤖 AskMyDocs by SHOEB")

# ✅ Access Together API key securely from Streamlit Secrets
api_key = st.secrets["general"]["API_KEY"]

# 📄 File uploader
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("📚 Analyzing your file..."):
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load and process the PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)

        # Embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)

        # Load TogetherAI LLM with secret API key using TogetherAPI
        llm = TogetherAPI(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.7,
            max_tokens=512,
            api_key=api_key
        )

        # Create QA chain
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ Document processed! Ask your question below:")

        # User question input
        question = st.text_input("❓ Ask something about the document:")
        if question:
            with st.spinner("🤔 Thinking..."):
                response = qa_chain.run(question)
                st.success("✅ Answer:")
                st.write(response)

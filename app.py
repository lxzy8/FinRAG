import streamlit as st
from rag_processor import start_ollama, process_pdf, setup_rag
import tempfile
import os

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# UI Configuration
st.set_page_config(
    page_title="Financial RAG Analyst",
    page_icon="📊",
    layout="centered"
)

# App Header
st.title("🔍 Financial Document Analyst")
st.markdown("""
Analyze SEC filings using AI. Upload a 10-K PDF and ask questions.
""")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    st.markdown("---")
    st.caption("💡 Sample questions:")
    st.caption("- What are the key risk factors?")
    st.caption("- Show me the revenue breakdown")
    st.caption("- What was the net income in 2023?")

# Initialize Ollama
if not start_ollama():
    st.error("Failed to start Ollama server. Please ensure Ollama is installed.")
    st.stop()

# Document Processing
if uploaded_file:
    with st.spinner("Processing document..."):
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # Process document
            st.session_state.vector_store = process_pdf(tmp_path)
            st.session_state.rag_chain = setup_rag(st.session_state.vector_store)
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
        finally:
            os.unlink(tmp_path)  # Clean up temp file

# Chat Interface
if st.session_state.rag_chain:
    st.divider()
    user_question = st.chat_input("Ask a question about the document...")
    
    if user_question:
        with st.spinner("Analyzing..."):
            response = st.session_state.rag_chain({"query": user_question})
        
        # Display conversation
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.chat_message("assistant"):
            st.write(response["result"])
            
            with st.expander("View sources"):
                for i, doc in enumerate(response["source_documents"][:3]):
                    st.caption(f"📄 Source {i+1} (Page {doc.metadata['page']+1})")
                    st.text(doc.page_content[:300] + "...")
else:
    st.info("Please upload a financial document to begin analysis")

# Footer
st.divider()
st.caption("Built with Ollama • LangChain • Streamlit | Model: deepseek-r1:1.5b")
import subprocess
import time
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def start_ollama():
    """Ensure Ollama server is running"""
    try:
        # Check if server is responsive
        requests.get("http://localhost:11434", timeout=2)
        return True
    except:
        try:
            # Start server if not running
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)  # Wait for server initialization
            return True
        except Exception as e:
            print(f"Ollama start failed: {e}")
            return False

def check_model():
    """Verify if required model is installed"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        return "deepseek-r1:1.5b" in result.stdout
    except:
        return False

def download_model():
    """Download the required Ollama model"""
    if not check_model():
        print("Downloading deepseek-r1:1.5b (1.1GB)...")
        subprocess.run(
            ["ollama", "pull", "deepseek-r1:1.5b"],
            check=True
        )

def process_pdf(file_path):
    """Process PDF into vector store"""
    # Load and split PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    
    # Configure text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    
    # Create chunks with metadata
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc.page_content)
        for text in split_texts:
            chunks.append({
                "text": text,
                "metadata": {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0),
                    "report_type": "10-K"
                }
            })
    
    # Initialize embeddings
    download_model()  # Ensure model is available
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    
    # Create vector store
    return Chroma.from_texts(
        texts=[c["text"] for c in chunks],
        embedding=embeddings,
        metadatas=[c["metadata"] for c in chunks],
        persist_directory="./chroma_db"
    )

def setup_rag(vector_store):
    """Configure RAG pipeline"""
    llm = Ollama(
        model="deepseek-r1:1.5b",
        temperature=0.3,
        top_k=50
    )
    
    # Custom prompt template
    template = """
    You are a financial analyst. Answer concisely using ONLY these facts:
    
    Context:
    {context}
    
    Question: {question}
    
    Respond in this format:
    - Key Answer: [1-2 sentence summary]
    - Supporting Evidence: [relevant excerpt]
    - Page Reference: [source page]
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
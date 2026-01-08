import os
import sys
import subprocess
import tempfile
import time

# --- DEPENDENCY MANAGEMENT ---
def install_dependencies():
    """Checks and installs necessary dependencies."""
    required_packages = [
        "langchain", 
        "langchain-community", 
        "chromadb", 
        "streamlit", 
        "pypdf",
        "sentence-transformers", 
        "langchain-huggingface", 
        "huggingface_hub",
        "accelerate" # Often needed for HF
    ]
    print("Checking system requirements...")
    for package in required_packages:
        try:
            # Simple check. For some (like langchain-community), import name differs from package name.
            # We trust pip to handle "requirement already satisfied".
            # We'll just run pip install for all if we suspect anything is missing, 
            # or to be safe, just run it. It's fast if cached.
            pass 
        except ImportError:
            pass
            
    # For robust sharing, just run pip install. 
    # It ensures everything is there without complex import checks mapping to package names.
    print("Ensuring dependencies are installed...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_packages)
    print("All dependencies checked/installed.")

# --- BOOTSTRAP LOGIC ---
if __name__ == "__main__":
    # Check if running via Streamlit by looking for specific env var we set
    if os.environ.get("NCERT_SOLVER_RUNNING") != "true":
        install_dependencies()
        print("Launching NCERT Doubt Solver...")
        env = os.environ.copy()
        env["NCERT_SOLVER_RUNNING"] = "true"
        try:
            # Launch streamlit
            subprocess.run(["streamlit", "run", __file__], env=env, check=True)
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error launching app: {e}")
            input("Press Enter to exit...")
        sys.exit()

# ==========================================
# APPLICATION CODE (Runs inside Streamlit)
# ==========================================

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# --- CONFIGURATION ---
try:
    st.set_page_config(page_title="NCERT Doubt Solver", layout="wide")
except:
    pass # In case something weird happens with double init

DB_DIR = "chroma_db_single" # Use a separate DB for this single file version

# --- INGEST MODULE ---
def load_pdf(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# --- VECTOR STORE MODULE ---
def get_embedding_function():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

def initialize_vector_store():
    embedding_function = get_embedding_function()
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_function
    )
    return vector_store

def add_documents_to_store(chunks):
    vector_store = initialize_vector_store()
    vector_store.add_documents(chunks)
    return vector_store

# --- LLM SERVICE MODULE ---
class LLMService:
    def __init__(self):
        # Integrated API Key (Provided by User)
        self.api_token = "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"
        
        # Using Qwen2.5-7B-Instruct for best Multilingual (Hindi/English) support
        self.repo_id = "Qwen/Qwen2.5-7B-Instruct"
        self.client = InferenceClient(token=self.api_token)

    def get_response(self, context: str, query: str) -> str:
        try:
            # Structured prompt strictly for Question Answering
            messages = [
                {"role": "user", "content": f"You are a helpful NCERT tutor. Answer this question using ONLY the context provided below. If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {query}"}
            ]
            
            response = self.client.chat_completion(
                messages,
                model=self.repo_id,
                max_tokens=512,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with AI: {e}"

# --- PIPELINE CLASS ---
class RAGPipeline:
    def __init__(self):
        self.vector_store = initialize_vector_store()
        self.llm_service = LLMService()

    def ingest_document(self, file_path: str):
        docs = load_pdf(file_path)
        chunks = chunk_documents(docs)
        add_documents_to_store(chunks)
        self.vector_store = initialize_vector_store()

    def query(self, question: str, k=3):
        results = self.vector_store.similarity_search(question, k=k)
        if not results:
            return "No relevant context found in the uploaded documents.", []
        
        context = "\n\n".join([doc.page_content for doc in results])
        answer = self.llm_service.get_response(context, question)
        return answer, results

# --- MAIN APP LOGIC ---
@st.cache_resource
def get_pipeline():
    return RAGPipeline()

def main():
    st.title("📚 Multilingual NCERT Doubt Solver (Professional)")
    st.markdown("Use this tool to upload your textbooks and ask questions in English or Hindi.")
    
    # Check if API token works or warn user
    # (Optional, but good for UX)

    pipeline = get_pipeline()

    # Sidebar
    with st.sidebar:
        st.header("Upload Textbook")
        uploaded_file = st.file_uploader("Upload a PDF Chapter", type=["pdf"])
        
        if uploaded_file is not None:
            # Safely create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            if st.button("Ingest Document"):
                with st.spinner("Processing Document..."):
                    try:
                        pipeline.ingest_document(tmp_path)
                        st.success("Document added to knowledge base!")
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    # Main Area
    st.header("Ask a doubt")
    question = st.text_input("Enter your question here (e.g., 'What are acids?' or 'अम्ल क्या हैं?'):")
    
    if st.button("Get Answer"):
        if question:
            with st.spinner("Analyzing..."):
                answer, results = pipeline.query(question)
                
                st.markdown("### Answer")
                st.write(answer)
                
                with st.expander("View Source Context"):
                    for i, doc in enumerate(results):
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(doc.page_content)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

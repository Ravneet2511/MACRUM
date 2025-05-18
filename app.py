import streamlit as st
import os
import torch
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.document_loaders import TextLoader
from llama_index.readers.file import DocxReader, PptxReader
import tempfile  # For handling uploaded files
from dotenv import load_dotenv

load_dotenv()

custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# --- Helper Functions ---
def get_pdf_text(pdf_files):
    text_chunks = []
    for pdf_file in pdf_files:
        filename = pdf_file.name
        pdf_reader = PdfReader(pdf_file)
        full_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text
        if full_text:
            text_chunks.append((full_text, filename))
    return text_chunks


def get_txt_text(txt_files):
    text_chunks = []
    for txt_file in txt_files:
        filename = txt_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(txt_file.getvalue())
            tmp_path = tmp_file.name
        try:
            with open(tmp_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(tmp_path, 'r', encoding='latin-1') as f:
                text = f.read()
        finally:
            os.remove(tmp_path)
        if text:
            text_chunks.append((text, filename))
    return text_chunks


def get_docx_text(docx_files):
    text_chunks = []
    for f in docx_files:
        filename = f.name
        docs = DocxReader(f).load_data()
        full_text = "\n".join(d.text for d in docs if hasattr(d, 'text'))
        if full_text:
            text_chunks.append((full_text, filename))
    return text_chunks


def get_pptx_text(pptx_files):
    text_chunks = []
    for f in pptx_files:
        filename = f.name
        docs = PptxReader(f).load_data()
        full_text = "\n".join(d.text for d in docs if hasattr(d, 'text'))
        if full_text:
            text_chunks.append((full_text, filename))
    return text_chunks


def get_chunks_with_metadata(text_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len
    )
    chunks, metadatas = [], []
    for text, filename in text_data:
        parts = splitter.split_text(text)
        for i, chunk in enumerate(parts):
            chunks.append(chunk)
            metadatas.append({
                "source": filename,
                "chunk": i,
                "chunk_id": f"{filename}_chunk_{i}"
            })
    return chunks, metadatas


def get_vectorstore(chunks, metadatas):
    if not chunks:
        st.error("No text chunks found to create a vector store.")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        return FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


def get_conversation_chain(vectorstore, groq_api_key, model_name):
    if not vectorstore or not groq_api_key:
        st.error("Vectorstore or GROQ API key missing.")
        return None
    llm = ChatGroq(
        api_key=groq_api_key,
        model=model_name,
        temperature=0.4,
        max_tokens=1024
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory,
        return_source_documents=True
    )

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="M.A.C.R.U.M", layout="wide", page_icon="📄")
    st.header("Chat with Your Documents using LLMs 🧠")

    # Session state init
    for key, default in {
        'conversation_chain': None,
        'chat_history': [],
        'vectorstore': None,
        'processed_text': None,
        'groq_api_key': ''
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.subheader("⚙️ Configuration")
        key_input = st.text_input("Enter your Groq API Key:", type="password", value=st.session_state.groq_api_key)
        if key_input: st.session_state.groq_api_key = key_input
        models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "deepseek-r1-distill-llama-70b",
            "qwen-qwq-32b",
        ]
        selected_model = st.selectbox("Select LLM Model:", models, index=0)
        st.subheader("📚 Your Documents")
        uploaded = st.file_uploader(
            "Upload PDF, TXT, DOCX, PPTX and click 'Process'", accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'pptx']
        )
        if st.button("Process Documents"):
            if not uploaded:
                st.warning("Please upload at least one document.")
            elif not st.session_state.groq_api_key:
                st.warning("Please enter your Groq API Key.")
            else:
                with st.spinner("Processing documents..."):
                    pdfs, txts, docs, ppts = [], [], [], []
                    for f in uploaded:
                        t = f.type
                        if t == "application/pdf": pdfs.append(f)
                        elif t == "text/plain": txts.append(f)
                        elif t.endswith("wordprocessingml.document"): docs.append(f)
                        elif t.endswith("presentationml.presentation"): ppts.append(f)

                    all_text_data = []
                    all_text_data.extend(get_pdf_text(pdfs))
                    all_text_data.extend(get_txt_text(txts))
                    all_text_data.extend(get_docx_text(docs))
                    all_text_data.extend(get_pptx_text(ppts))

                    if not all_text_data:
                        st.error("No text could be extracted from the uploaded documents.")
                        return

                    chunks, metadatas = get_chunks_with_metadata(all_text_data)
                    vs = get_vectorstore(chunks, metadatas)
                    if vs:
                        st.session_state.vectorstore = vs
                        st.session_state.conversation_chain = get_conversation_chain(
                            vs, st.session_state.groq_api_key, selected_model
                        )
                        st.success("Documents processed successfully! You can now ask questions.")
                        st.session_state.chat_history = []
                    else:
                        st.error("Failed to create vector store.")

        if st.session_state.vectorstore and st.button("Clear Processed Data & Chat"):
            for k in ['vectorstore', 'conversation_chain', 'chat_history', 'processed_text']:
                st.session_state[k] = None if k!='chat_history' else []
            st.info("Cleared. Upload new documents or re-process.")
            st.experimental_rerun()

    # Chat display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View Sources"):
                    for i, doc in enumerate(msg["sources"]):
                        fn = doc.metadata.get('source', 'Unknown')
                        ch = doc.metadata.get('chunk', 'N/A')
                        st.write(f"**Source {i+1}: {fn} (Chunk {ch})**")
                        st.caption(doc.page_content[:300] + "...")

    # Chat input & ask
    user_q = st.chat_input("Ask a question about your documents:")
    if user_q:
        if not st.session_state.conversation_chain:
            st.warning("Process documents first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"): st.markdown(user_q)
            with st.spinner("Thinking..."):
                try:
                    res = st.session_state.conversation_chain({"question": user_q})
                    ans, srcs = res["answer"], res.get("source_documents", [])
                    st.session_state.chat_history.append({"role":"assistant","content":ans,"sources":srcs})
                    st.session_state.chat_history = st.session_state.chat_history[-10:]
                    with st.chat_message("assistant"): 
                        st.markdown(ans)
                        with st.expander("View Sources"):
                            for i, d in enumerate(srcs):
                                fn = d.metadata.get('source','Unknown')
                                ch = d.metadata.get('chunk','N/A')
                                st.write(f"**Source {i+1}: {fn} (Chunk {ch})**")
                                st.caption(d.page_content[:300]+"...")
                except Exception as e:
                    st.error(f"Conversation error: {e}")

    # Download chat
    if st.session_state.chat_history:
        text = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.chat_history)
        st.download_button("Export Chat", data=text, file_name="chat_history.txt")

if __name__ == "__main__":
    main()

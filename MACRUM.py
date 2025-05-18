import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from groq import Groq
from langchain.document_loaders import TextLoader
# from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import DocxReader,PptxReader

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

def get_pdf_text(docs):
    text=""
    for pdf in docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_txt_text(files):
    text = ""
    for path in files:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()  # returns a list of 1‐element Documents
        for doc in docs:
            text += doc.page_content + "\n"
    return text

def get_docx_text(docx_paths):
    text = ""
    for path in docx_paths:
        # use DocxReader on each file
        reader = DocxReader()
        docs = reader.load_data(path)  # pass list of one path
        for doc in docs:
            text += doc.text + "\n"
    return text

def get_pptx_text(pptx_paths):
    text = ""
    for path in pptx_paths:
        reader = PptxReader()
        docs = reader.load_data([path])  # pass a list of one .pptx path
        for doc in docs:
            text += doc.text + "\n"
    return text

# converting text to chunks
def get_chunks(raw_text):
    text_splitter=RecursiveCharacterTextSplitter(
                                        # separator="\n",
                                        chunk_size=1024,
                                        chunk_overlap=200,
                                        length_function=len
                                        )   
    chunks=text_splitter.split_text(raw_text)
    return chunks

# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunks):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    vectorstore=faiss.FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))])
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=1024,
    )
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') # using conversation buffer memory to hold past information
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory,
                                return_source_documents=True 
                                )
    return conversation_chain

def get_all_uploaded_filepaths(upload_dir=r"C:\\Users\\LENOVO\\OneDrive\\Desktop\\MARUM\\"):
    # Get all file paths in the upload directory
    return [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]

def main():
    print("Scanning uploaded files in /content ...")
    raw = ""

    filepaths = get_all_uploaded_filepaths()

    for filepath in filepaths:
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == ".pdf":
                print(f"Processing PDF: {filepath}")
                raw += get_pdf_text([filepath])
            elif ext == ".txt":
                print(f"Processing TXT: {filepath}")
                raw += get_txt_text([filepath])
            elif ext == ".docx":
                print(f"Processing DOCX: {filepath}")
                raw += get_docx_text([filepath])
            elif ext == ".pptx":
                print(f"Processing PPTX: {filepath}")
                raw += get_pptx_text([filepath])
            else:
                print(f"Skipping unsupported file: {filepath}")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if not raw.strip():
        print("No valid documents found. Exiting.")
        return

    chunks = get_chunks(raw)
    vectorstore = get_vectorstore(chunks)
    # use the same name here as in the loop below:
    conversation_chain = get_conversation_chain(vectorstore)

    print("\nDocument processing complete. Ask your questions below. Type 'bye' to exit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Session ended. Have a great day!")
            break

        # Call the correct variable
        result = conversation_chain({"question": user_input})
        answer = result["answer"]
        source_docs = result.get("source_documents", [])

        # Filter only docs that actually have your metadata key
        valid_sources = [
            doc for doc in source_docs
            if doc.metadata and doc.metadata.get("source_file")
        ]

        # Print the answer
        print("AI:", answer)

        # Only print the Sources block if there's at least one valid source
        if valid_sources:
            print("\nSources:")
            for doc in valid_sources:
                meta = doc.metadata
                print(f" • {meta['source_file']} (page {meta.get('page', '?')}, chunk #{meta.get('chunk_index', '?')})")
                snippet = doc.page_content.replace("\n", " ")[:200]
                print(f"   → “{snippet}…”")

if __name__ == "__main__":
    main()
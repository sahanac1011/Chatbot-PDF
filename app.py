import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.base import Embeddings
import os
from typing import List, Dict
import chromadb
import voyageai

api_key = os.environ.get("ANTHROPIC_API_KEY")

class PDFChatbot:
    def __init__(self):
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'dimensionality': 128}
        )
        
        os.makedirs("chroma_db", exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        
        # Initialize Anthropic LLM
        self.llm = ChatAnthropic(
            api_key=api_key, # type: ignore
            model_name="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens_to_sample=1000,
            timeout=None,
            stop= None
            
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.vector_store = None
        self.qa_chain = None
        self.collection_name = "pdf_collection"

    def load_pdf(self, pdf_file: str) -> None:
        """Load and process a PDF file"""
        # Load PDF
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create or get collection
        try:
            self.chroma_client.create_collection(self.collection_name)
        except:
            pass
        
        # Create vector store with ChromaDB
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name=self.collection_name
        )
        
        # Create conversational QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )

    def ask_question(self, question: str) -> Dict:
        """Process a question and return an answer with sources"""
        if not self.qa_chain:
            return {
                "answer": "Please load a PDF document first.",
                "sources": []
            }
        
        try:
            # Get response from QA chain
            response = self.qa_chain({"question": question})
            
            # Extract source documents
            sources = []
            if response.get("source_documents"):
                sources = [doc.page_content[:200] + "..." for doc in response["source_documents"]]
            
            return {
                "answer": response["answer"],
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }

def main():
    st.title("PDF Analysis Chatbot")
    
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state or st.sidebar.button("Reset Chatbot"):
        st.session_state.chatbot = PDFChatbot()
    
    # File upload
    pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if pdf_file:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                st.session_state.chatbot.load_pdf("temp.pdf")
            st.success("PDF processed successfully!")
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Question input
    question = st.chat_input("Ask a question about the PDF:")
    if question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Get response
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.ask_question(question)
            
            # Format answer with sources
            answer_text = response["answer"]
            if response["sources"]:
                answer_text += "\n\nSources:"
                for idx, source in enumerate(response["sources"], 1):
                    answer_text += f"\n{idx}. {source}"
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
        
        # Rerun to update chat display
        st.rerun()

if __name__ == "__main__":
    main()
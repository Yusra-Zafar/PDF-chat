import os                                
from dotenv import load_dotenv          
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai


# configuring API key
load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# extracting text from all the pages of all the pdf documents uploaded by the user
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:  # looping through pdfs
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:  # looping through pages
            text += page.extract_text()
    return text

# splitting text into chunks for easy storage and retrieval and passing to LLM (limited context window)
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000, chunk_overlap = 1000
        )    
    chunks = text_splitter.split_text(text = text)
    return chunks

# converting these chunks into vector embeddings, using google embeddings model, faiss vector store and storing locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")     # creating vectore store for the documents, from the embeddings
    db = FAISS.from_texts(texts = text_chunks, embedding = embeddings, )    # ERROR: used from_documents instead of from_texts :[
    db.save_local("faiss_index")

# setting up the prompt template and chain for conversation with the LLM
def get_conversational_chain():
    prompt_template = """
    You are a helpful question answering assistant using documents as the context. 
    You will be given source documents and a user's question. 
    You have to read, understand and analyze the question and provide the best suited answer. 
    Only respond to relevant questions.
    DO NOT make up answer by yourself.
    Say "I don't know" if you do not know something.
    Maintain polite, humble, professional and friendly tone.

    {context}
    Question: {question}
    Helpful Answer:
    """

    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)
    prompt = PromptTemplate(input_variables= ["context", "question"], template= prompt_template)
    chain = load_qa_chain(llm = model, chain_type = "stuff", prompt = prompt)

    return chain

# process user query, search in the db and return response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")    # create embeddings
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization= True)    # load local vector store
    docs = new_db.similarity_search(user_question)        # perform similarity search of question with the vector store
    
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

# building streamlit UI
def main():
    st.set_page_config("Chat with PDFs")
    st.title("Chat with multiple PDFs using Gemini")

    user_question = st.text_input("Ask your question", placeholder="Ask from the documents")    # ERROR: on_change:True

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDFs here", type="pdf", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()


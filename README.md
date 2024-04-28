# Chat with PDFs using Gemini

The "Chat with PDFs using Gemini" project is a streamlit application that allows users to ask questions based on multiple PDF documents uploaded by the user. 
It utilizes Google Generative AI (Gemini) for document understanding and question answering.

![pdf_chat](https://github.com/Yusra-Zafar/PDF-chat/assets/141744510/e1a20956-a8ed-4643-a1b1-a3267490e98e)


## Features

- Extracts text from multiple PDF documents uploaded by the user.
- Splits the text into manageable chunks for efficient processing.
- Converts text chunks into vector embeddings using Google Generative AI embeddings.
- Stores vector embeddings locally for faster retrieval.
- Provides a conversational interface for users to ask questions related to the uploaded documents.
- Maintains a polite, humble, professional, and friendly tone in responses.

## Usage

1. Upload multiple PDF documents using the file uploader in the sidebar.
2. Enter your question related to the uploaded documents in the text input box.
3. Click "Submit" to process the question and receive a response.
4. The application will provide a helpful answer based on the context of the uploaded documents.

## Dependencies

- `os`: Provides a way to interact with the operating system.
- `dotenv`: Python library for managing environment variables.
- `streamlit`: Python library for building web applications.
- `PyPDF2`: Python library for reading PDF files.
- `langchain`: Library for text processing and question answering.
- `google.generativeai`: Library for accessing the Gemini AI model.

## How to Run

1. Clone the repository.
2. Install required dependencies using `pip install -r requirements.txt`.
3. Ensure you have the required environment variables set up (e.g., GOOGLE_API_KEY for Gemini).
4. Run the application using `streamlit run chat_with_pdfs.py`.
5. Upload PDF documents and start asking questions related to the documents.


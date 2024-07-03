# SemanticRetrieval-RAG-using-LangChain & OpenAI

LangChain-RAG is a project that implements Retrieval-Augmented Generation (RAG) using LangChain and OpenAI's language models. This project demonstrates how to use LangChain for document retrieval and embedding, and OpenAI's API for generating responses based on the retrieved documents from **Websites**. 

## Project Structure

- `package_requirements/`:
   - `install_requirements.sh/`:
   - `requirements.txt/`
     
- `configuration.py`: Contains the Config class to set up environment variables.
- `document_processor.py`: Handles loading, splitting, and embedding documents.
- `rag_process.py`: Contains the RAGProcess class for creating and querying the RAG model.
- `setup.py`: Script to install required packages.
- `main.py`: Main script to run the RAG process.

### Usage

To run the RAG process, use the run.py script. This script will **prompt you to enter a question and then use the RAG model to generate a response**.

The script will:

- Set up the configuration using the provided API keys.
- Load and preprocess documents from the specified URL.
- Generate a response based on the provided question.
- Calculate and display the cosine similarity score between the query and the retrieved document.


from configuration import Config
from rag import RAGProcess

def main():
    
    """
    Executes the main function to run the RAG process.
    
    parameters
    ----------
    None
    
    Process
    -------
    1. Set the API keys.
    2. Run the RAG process.
    3. Set the parameters for the RAG process.
    4. Return the response from the model.
    5. Calculate the cosine similarity score and return it.
    
    """
    
    api_key = "your_langchain_api_key"  
    openai_api_key = "your_openai_api_key"  
    config = Config(api_key, openai_api_key)

    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    # Run the RAG process
    rag_process = RAGProcess()
    
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    parse_classes = ["post-content", "post-title", "post-header"]
    question = input("Enter your question: ")
    response, retriever = rag_process.run(url, parse_classes, question)
    print("RAG Response:", response)
    
    # Calculate cosine similarity
    documents = retriever.retrieve(query=question, n_results=1)
    document = documents[0]
    similarity_score = rag_process.calculate_cosine_similarity(question, document.page_content)
    print("Cosine Similarity Score:", similarity_score)

if __name__ == "__main__":
    main()

from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import numpy as np

from document_processor import DocumentProcessor

class RAGProcess:
    """
    This class defines the RAG process to query a document and calculate cosine similarity.
    
    Attributes:
    - openai_api_key: The OpenAI API key.
    
    Methods
    -------
    create_rag_chain:
        Create a RAG chain.
        
    query_rag_chain:
        Query the RAG chain.
        
    calculate_cosine_similarity:
        Calculate the cosine similarity between the query and the document.
        
    run:
        Run the RAG process.   
    """
    def __init__(self):
        """
        initialise the RAGProcess class.
        
        parameters
        ----------
        None
        
        attributes
        ----------
        openai_api_key:
            The OpenAI API key.
            
        Raises
        -------
        ValueError:
            If the OPENAI_API_KEY environment variable is not set.
        
        """
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        
        self.doc_processor = DocumentProcessor(self.openai_api_key)


    def create_rag_chain(self, retriever):
        """
        Create a RAG chain.
        
        parameters
        ----------
        retriever:
            The retrieve information to use.
            
        Implements
        ----------
        hub.pull:
            Pull a chain from the hub.
            
        ChatOpenAI:
            A chat model that uses the OpenAI API.
            
        StrOutputParser:
            A parser that parses the output as a string.
            
        Raises
        -------
        ValueError:
            If the retriever is empty.
            If failed to create a RAG chain.

        """
        if not retriever:
            raise ValueError("Retriever cannot be empty.")
        
        try:
            prompt = hub.pull("rlm/rag-prompt")
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            rag_chain = (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            if not rag_chain:
                raise ValueError("Failed to create RAG chain.")
            return rag_chain
        except Exception as e:
            raise ValueError(f"Failed to create RAG chain: {str(e)}")

    def query_rag_chain(self, rag_chain, question):
        """
        Query the RAG chain.
        
        parameters
        ----------
        rag_chain:
            The RAG chain to query.
            
        question:
            The question to ask.
            
        Raises
        -------
        ValueError:
            If the RAG chain is empty.
            If the question is empty.
            If failed to get a response from the RAG chain.
            If failed to query the RAG chain.
        """
        if not rag_chain:
            raise ValueError("RAG chain cannot be empty.")
        if not question:
            raise ValueError("Question cannot be empty.")
        
        try:
            response = rag_chain.invoke(question)
            if not response:
                raise ValueError("Failed to get a response from the RAG chain.")
            return response
        except Exception as e:
            raise ValueError(f"Failed to query RAG chain: {str(e)}")

    def calculate_cosine_similarity(self, query, document):
        """
        Calculate the cosine similarity between the query and the document.
        
        parameters
        ----------
        query:
            The query to compare.
            
        document:
            The document to compare.
            
        Implements
        ----------
        OpenAIEmbeddings:
            An embedding model that uses the OpenAI API to embed the query and document.
            
        Raises
        -------
        ValueError:
            If the query is empty.
            If the document is empty.
            If failed to calculate cosine similarity.
        """
        if not query:
            raise ValueError("Query cannot be empty.")
        if not document:
            raise ValueError("Document cannot be empty.")
        
        try:
            embd = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            query_result = embd.embed_query(query)
            document_result = embd.embed_query(document)
            
            def cosine_similarity(vec1, vec2):
                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                return dot_product / (norm_vec1 * norm_vec2)
            
            similarity_score = cosine_similarity(query_result, document_result)
            return similarity_score
        except Exception as e:
            raise ValueError(f"Failed to calculate cosine similarity: {str(e)}")

    def run(self, url, parse_classes, question):
        """
        Run the RAG process.  

        parameters
        ----------
        url:
            Loads the URL for processing the document.

        parse_classes:
            Parses the classes for processing the document.

        question:
            Query from the user's prompt. 
        
        """
        try:
            docs = self.doc_processor.load_documents(url, parse_classes)
            splits = self.doc_processor.split_documents(docs)
            retriever = self.doc_processor.embed_documents(splits)
            rag_chain = self.create_rag_chain(retriever)
            response = self.query_rag_chain(rag_chain, question)
            return response, retriever
        except Exception as e:
            raise ValueError(f"Failed to run the RAG process: {str(e)}")



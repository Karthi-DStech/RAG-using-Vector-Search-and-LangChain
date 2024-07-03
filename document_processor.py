from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import bs4
import os


class DocumentProcessor:
    """
    This class defines the document processor to load, split, embed, and format documents.
    
    Attributes
    ----------
    openai_api_key: 
        The OpenAI API key.
        
    Methods
    -------
    load_documents:
        Load documents from the provided URL.
        
    split_documents:
        Split the documents into chunks.
        
    embed_documents:
        Embed the documents.
    """
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def load_documents(self, url, parse_classes):
        """
        Load documents from the provided URL.

        parameters
        ----------
        url:
            The URL to load documents from.

        parse_classes:
            The parse classes to use.

        Implements
        ----------
        WebBaseLoader:
            A document loader that loads documents from the web.

        bs4:
            A library for parsing HTML and XML documents.

        Raises
        -------
        ValueError:
            If the URL is empty.
            If the parse classes are empty.
            If no documents are found at the provided URL.
            If failed to load documents from the URL.
        """
        if not url:
            raise ValueError("URL cannot be empty.")
        if not parse_classes:
            raise ValueError("Parse classes cannot be empty.")

        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(class_=tuple(parse_classes))
                ),
            )
            documents = loader.load()
            if not documents:
                raise ValueError("No documents found at the provided URL.")
            return documents
        except Exception as e:
            raise ValueError(f"Failed to load documents from the URL: {str(e)}")

    def split_documents(self, docs, chunk_size=1000, chunk_overlap=200):
        """
        Split the documents into chunks (preprocessing).

        parameters
        ----------
        docs:
            The documents to split.

        chunk_size:
            Specifies the maximum size of each chunk in terms of characters

        chunk_overlap:
            Specifies the number of characters that overlap between consecutive chunks.

        Implements
        ----------
        RecursiveCharacterTextSplitter:
            A text splitter that splits documents into chunks.

        Raises
        -------
        ValueError:
            If the documents are empty.
            If failed to split documents into chunks.
        """
        if not docs:
            raise ValueError("Documents cannot be empty.")

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            splits = text_splitter.split_documents(docs)
            if not splits:
                raise ValueError("Failed to split documents into chunks.")
            return splits
        except Exception as e:
            raise ValueError(f"Failed to split documents: {str(e)}")

    def embed_documents(self, splits):
        """
        Embed the documents.

        parameters
        ----------
        splits:
            The splits to embed.

        Implements
        ----------
        Chroma:
            A vectorstore that stores document embeddings.

        OpenAIEmbeddings:
            An embedding model that uses the OpenAI API.

        Raises
        -------
        ValueError:
            If the splits are empty.
            If failed to embed documents.
        """
        if not splits:
            raise ValueError("Splits cannot be empty.")

        try:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=OpenAIEmbeddings(openai_api_key=self.openai_api_key),
            )
            retriever = vectorstore.as_retriever()
            if not retriever:
                raise ValueError("Failed to create retriever from document embeddings.")
            return retriever
        except Exception as e:
            raise ValueError(f"Failed to embed documents: {str(e)}")

    def format_docs(self, docs):
        """
        Format the documents.

        parameters
        ----------
        docs:
            The documents to format.

        Implements
        ----------
        page_content:
            The content of the document.

        Raises
        -------
        ValueError:
            If the documents are empty.
            If failed to format documents.
        """
        if not docs:
            raise ValueError("Documents cannot be empty.")

        try:
            formatted_docs = "\n\n".join(doc.page_content for doc in docs)
            if not formatted_docs:
                raise ValueError("Failed to format documents.")
            return formatted_docs
        except Exception as e:
            raise ValueError(f"Failed to format documents: {str(e)}")

class Config:
    """
    The Config class sets the API keys for the RAG process.

    Attributes
    ----------
    api_key:
        The Langchain API key.

    Methods
    -------
    setup_environment:
        Set up the environment for the RAG process.

    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.setup_environment()

    def setup_environment(self):
        import os

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = self.api_key
        os.environ["OPENAI_API_KEY"] = self.api_key

from src.core.retriever import WebRetriever

class Tool:
    """
    Base class fro any tool the agent might use.
    """
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input_text):
        return self.func(input_text)
    
class SearchTool(Tool):
    """
    Wrapper specifically for our WebRetriever.
    Standarizes the input/output for the agent.
    """

    def __init__(self, max_results=1):
        # Instanciamos el retriever robusto que creamos en el Milestone anterior
        self.retriever = WebRetriever(max_results=max_results)
        
        super().__init__(
            name="Search",
            description="Useful for finding facts, dates, events, and verifying specific details about entities. Input should be a specific search query.",
            func=self.retriever.search
        )

if __name__ == "__main__":
    tool = SearchTool()
    print(f"Tool Name: {tool.name}")
    print(f"Tool Description: {tool.description}")
    print(f"Test Run: {tool.run('Python programming language creator')}")
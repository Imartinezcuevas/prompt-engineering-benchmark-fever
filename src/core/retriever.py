# src/core/retriever.py
from ddgs import DDGS
from googlesearch import search as google_search
import wikipedia

class WebRetriever:
    """
    A multi-stage retriever that attempts to fetch evidence from:
    1. DuckDuckGo (Fastest)
    2. Google Search (Broadest)
    3. Wikipedia (Most reliable for encyclopedic facts)
    """
    
    def __init__(self, max_results=1):
        self.max_results = max_results
        self.ddgs = DDGS()
        wikipedia.set_lang("en")

    def search_duckduckgo(self, query):
        """Try searching with DuckDuckGo."""
        try:
            results = self.ddgs.text(query, max_results=self.max_results, backend='auto')
            
            if not results:
                return None
                
            evidence_list = []
            for i, res in enumerate(results, 1):
                body = res.get('body', '')
                if body:
                    evidence_list.append(f"[Source DDG-{i}]: {body}")
            return "\n".join(evidence_list)
            
        except Exception as e:
            print(f" DuckDuckGo failed: {e}")
            return None

    def search_google(self, query):
        """Fallback 1: Try searching with Google."""
        try:
            results = google_search(query, num_results=self.max_results, advanced=True)
            
            evidence_list = []
            results = list(results) # Convert generator to list
            
            if not results:
                return None
                
            for i, res in enumerate(results, 1):
                body = res.description
                if body:
                    evidence_list.append(f"[Source Google-{i}]: {body}")
            return "\n".join(evidence_list)
            
        except Exception as e:
            print(f" Google failed: {e}")
            return None

    def search_wikipedia(self, query):
        """Fallback 2: Try searching official Wikipedia API."""
        try:
            # Clean query: Remove "fact check" instructions for better Wiki matching
            clean_query = query.replace(" fact check", "").replace("full cast", "").strip()
            
            search_results = wikipedia.search(clean_query)
            if not search_results:
                return None
            
            # 2. SANITY CHECK: Does the page title match our query keywords?
            best_page_title = search_results[0]
            
            # Simple fuzzy check: Do words from the title appear in our query?
            query_words = set(clean_query.lower().split())
            title_words = set(best_page_title.lower().split())
            
            # Intersection check (at least one significant word must match)
            if not query_words.intersection(title_words) and len(query_words) > 0:
                 # Try the second result
                 if len(search_results) > 1:
                     best_page_title = search_results[1]
                 else:
                     return None

            # 3. Get summary
            summary = wikipedia.summary(best_page_title, sentences=10, auto_suggest=False)
            
            return f"[Source Wikipedia - {best_page_title}]: {summary}"
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle ambiguous terms (e.g. "Mercury" -> Planet or Element?)
            try:
                first_option = e.options[0]
                summary = wikipedia.summary(first_option, sentences=3)
                return f"[Source Wikipedia - {first_option}]: {summary}"
            except:
                return None
        except Exception as e:
            print(f" Wikipedia failed: {e}")
            return None

    def search(self, query):
        """
        Main orchestration method.
        """
        # Append 'fact check' for search engines, but keep raw query for Wikipedia
        search_engine_query = f"{query} fact check"

        # 1. Try Wikipedia
        evidence = self.search_wikipedia(query)
        if evidence:
            return evidence
        
        # 2. Fallback to DuckDuckGo
        print("Switching to DuckDuckGo fallback...")
        evidence = self.search_duckduckgo(search_engine_query)
        if evidence:
            return evidence
            
        # 3. Fallback to Google
        print("Switching to Google Search fallback...")
        evidence = self.search_google(search_engine_query)
        if evidence:
            return evidence
            
        return "No external evidence found."

if __name__ == "__main__":
    print("Testing Robust WebRetriever Chain...")
    retriever = WebRetriever(max_results=1)
    
    # Test query
    query = "Cristiano Ronaldo ballon d'or count"
    print(f"Query: {query}")
    
    result = retriever.search(query)
    print(f"\nResult:\n{result}")
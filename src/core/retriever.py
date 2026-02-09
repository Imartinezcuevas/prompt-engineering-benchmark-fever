from ddgs import DDGS
from googlesearch import search as google_search
import wikipedia
import time

class WebRetriever:
    """
    A multi-stage retriever that attempts to fetch evidence from:
    1. DuckDuckGo (Fastest)
    2. Wikipedia (Most reliable for encyclopedic facts)
    3. Google Search (Broadest - last resort)
    """
    
    def __init__(self, max_results=2, verbose=False):
        self.max_results = max_results
        self.ddgs = DDGS()
        self.verbose = verbose
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
                    evidence_list.append(f"[DDG-{i}]: {body}")
            
            if evidence_list:
                return "\n".join(evidence_list)
            return None
            
        except Exception as e:
            if self.verbose:
                print(f" DuckDuckGo failed: {e}")
            return None

    def search_google(self, query):
        """Fallback: Try searching with Google."""
        try:
            results = google_search(query, num_results=self.max_results, advanced=True)
            
            evidence_list = []
            results = list(results)
            
            if not results:
                return None
                
            for i, res in enumerate(results, 1):
                body = res.description
                if body:
                    evidence_list.append(f"[Google-{i}]: {body}")
            
            if evidence_list:
                return "\n".join(evidence_list)
            return None
            
        except Exception as e:
            if self.verbose:
                print(f" Google failed: {e}")
            return None

    def search_wikipedia(self, query):
        """Try searching official Wikipedia API."""
        try:
            # Clean query: Remove "fact check" and extract key entities
            clean_query = (query.replace(" fact check", "")
                              .replace("full cast", "")
                              .replace("professional affiliations", "")
                              .strip())
            
            search_results = wikipedia.search(clean_query, results=5)
            if not search_results:
                return None

            # Find best matching page
            best_page = None
            
            # First pass: exact or close matches
            for title in search_results:
                if title.startswith("List of") or "discography" in title.lower():
                    continue
                
                if clean_query.lower() in title.lower() or title.lower() in clean_query.lower():
                    best_page = title
                    break
            
            # Second pass: first non-list page
            if not best_page:
                for title in search_results:
                    if not title.startswith("List of"):
                        best_page = title
                        break
            
            if not best_page:
                return None
            
            # Get summary
            summary = wikipedia.summary(best_page, sentences=5, auto_suggest=False)
            return f"[Wikipedia - {best_page}]: {summary}"
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle ambiguous terms
            try:
                first_option = e.options[0]
                summary = wikipedia.summary(first_option, sentences=3, auto_suggest=False)
                return f"[Wikipedia - {first_option}]: {summary}"
            except:
                return None
        except Exception as e:
            if self.verbose:
                print(f" Wikipedia failed: {e}")
            return None

    def search(self, query):
        """
        Main orchestration method with smart fallback strategy.
        """
        # Strategy 2: Try Wikipedia (better for entities/people)
        evidence = self.search_wikipedia(query)
        if evidence:
            return evidence
        
        # Strategy 1: Try DuckDuckGo with "fact check"
        evidence = self.search_duckduckgo(f"{query} fact check")
        if evidence:
            return evidence
        
        # Strategy 3: Try DuckDuckGo without "fact check"
        evidence = self.search_duckduckgo(query)
        if evidence:
            return evidence
        
        # Strategy 4: Last resort - Google
        if self.verbose:
            print("  → Using Google fallback...")
        evidence = self.search_google(f"{query} fact check")
        if evidence:
            return evidence
            
        return "No external evidence found for this query."


if __name__ == "__main__":
    print("Testing Improved WebRetriever...")
    retriever = WebRetriever(max_results=2, verbose=True)
    
    # Test queries
    test_queries = [
        "Nikolaj Coster-Waldau Fox Broadcasting",
        "Cristiano Ronaldo ballon d'or count",
        "Python programming language creator"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        result = retriever.search(query)
        print(f"Result:\n{result[:300]}...")
        time.sleep(2)
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from neuroform.tools.manager import tool_registry

def duckduckgo_search(query: str, max_results: int = 3) -> str:
    """Performs a web search using DuckDuckGo HTML and returns snippets."""
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()
            
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        for a in soup.find_all('a', class_='result__snippet', limit=max_results):
            text = a.get_text(strip=True)
            href = a.get('href', '')
            results.append(f"- {text}\n  URL: {href}")
            
        if not results:
            return "No results found."
            
        return "\n\n".join(results)
    except Exception as e:
        return f"Search error: {str(e)}"

def extract_webpage_text(url: str) -> str:
    """Fetches a URL and extracts the main text content."""
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove noisy tags
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        # Collapse multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        if len(text) > 10000:
            text = text[:10000] + "...\n[Content truncated for length]"
            
        return text
    except Exception as e:
        return f"Fetch error: {str(e)}"

# Register tools
tool_registry.register(
    func=duckduckgo_search,
    description="Search the live web using DuckDuckGo. Best for answering questions about current events, code, or general knowledge.",
    parameters={
        "query": {"type": "string", "description": "The web search query"}
    }
)

tool_registry.register(
    func=extract_webpage_text,
    description="Fetch the readable text content of a specific URL.",
    parameters={
        "url": {"type": "string", "description": "The full HTTP/HTTPS URL to read"}
    }
)

import json
from fastmcp import FastMCP
from web_scrap import get_news_events,get_notifications

from vector_db import collection

mcp = FastMCP("MCP for BMS College of Engineering")

@mcp.tool()
def get_latest_news():
    """
    Extracts the 'News & Events' Website,
    and returns the data as a JSON string.
    """ 
    return get_news_events()


@mcp.tool()
def get_college_notifications():
    """
    Extracts 'College Notifications' from the Website,
    and returns the data as a JSON string.
    """
    return get_notifications()


@mcp.tool()
def query_knowledge_base(query_text: str, n_results: int = 3) -> str:
    """
    Queries the ChromaDB vector store to find the most relevant document chunks for a given text query.
    """
    if not collection:
        return json.dumps({"error": "Cannot query. ChromaDB collection is not available."})
    
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        # CRITICAL CHANGE: Return the actual documents found, formatted as JSON.
        return json.dumps(results['documents'][0], indent=2)
    except Exception as e:
        return json.dumps({"error": f"An error occurred during the query: {e}"})

    

    
    




if __name__ == "__main__":
    mcp.run()
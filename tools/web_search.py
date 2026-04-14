import requests
import json

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo's free JSON API.
    Returns a list of {title, url, snippet} dicts.
    """
    url = 'https://api.duckduckgo.com/'

    params = {'q': query,
              'format': 'json',
              'no_redirect': 1,
              'no_html': 1, }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    results = []
    for item in data.get('RelatedTopics', [])[:max_results]:
        if 'Text' in item and 'FirstURL' in item:
            results.append({
                'title': item['Text'][:80],
                'url': item['FirstURL'],
                'snippet': item['Text']
         })

    return results
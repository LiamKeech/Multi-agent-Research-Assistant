from datetime import date
def format_citation(url: str, title: str, author: str = 'Unknown') -> str:
    """
    Formats a basic APA-style citation.
    """
    today = date.today()
    year = today.year
    accessed = today.strftime('%B %d, %Y')
    return (
        f'{author}. ({year}). {title}. ' 
        f'Retrieved {accessed}, from {url}'
    )
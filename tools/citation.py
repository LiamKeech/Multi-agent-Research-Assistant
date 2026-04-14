from datetime import date
from urllib.parse import urlparse


def _author_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith('www.'):
        host = host[4:]
    if not host:
        return 'Unknown'
    # Use main domain label as a simple source author fallback.
    label = host.split('.')[0].replace('-', ' ').strip()
    return label.title() if label else 'Unknown'


def format_citation(url: str, title: str, author: str | None = None) -> str:
    """
    Formats a basic APA-style citation.
    Falls back to a source name inferred from URL when author is missing.
    """
    today = date.today()
    year = today.year
    accessed = today.strftime('%B %d, %Y')
    resolved_author = (author or '').strip() or _author_from_url(url)
    return (
        f'{resolved_author}. ({year}). {title}. '
        f'Retrieved {accessed}, from {url}'
    )

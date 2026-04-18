import requests

OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'phi3'


def summarise_text(text: str, model: str = MODEL, ollama_url: str = OLLAMA_URL) -> str:
    """Send raw text to Ollama and return a concise summary."""
    cleaned_text = (text or '').strip()
    if not cleaned_text:
        return 'No text provided to summarise.'

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You are a research summarisation expert. '
                    'Condense the provided text into clear, concise bullet points.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    'Summarise the following text into 3-5 bullet points. '
                    'Return only the summary:\n\n'
                    f'{cleaned_text}'
                ),
            },
        ],
        'stream': False,
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get('message', {}).get('content', '').strip() or 'No summary could be generated.'
    except (requests.RequestException, ValueError) as exc:
        return f'Error contacting Ollama: {exc}'

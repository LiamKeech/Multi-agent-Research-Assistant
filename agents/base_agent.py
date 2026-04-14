import requests
import json

OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'phi3'

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: list[dict]):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools

    def _chat(self, messages: list[dict]) -> str:
        payload = {
            'model': MODEL,
            'messages': messages,
            'stream': False,
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()  # catches non-200 HTTP responses
            return response.json()['message']['content']

        except requests.exceptions.ConnectionError:
            return (
                'I could not connect to Ollama'
                'Please make sure it is running with: ollama serve'
            )
        except requests.exceptions.Timeout:
            return (
                'The request to Ollama timed out. '
                'The model may be too large for your machine — '
                'try switching to a smaller model like phi3 in base_agent.py.'
            )
        except requests.exceptions.RequestException as e:
            return f'An unexpected error occurred: {e}'

    def _build_tool_prompt(self) -> str:
        """Describe tools to the model in the system prompt."""
        if not self.tools:
            return ''
        lines = ['\n\nYou have access to the following tools.',
                 'To call a tool, respond ONLY with valid JSON in this format:',
                 '{"tool": "<tool_name>", "args": {<key: value>}}',
                 'Available tools:']
        for t in self.tools:
            lines.append(f' - {t["name"]}: {t["description"]}')
            lines.append(f' Parameters: {t["params"]}')
        return '\n'.join(lines)

    def _parse_tool_call(self, text: str) -> dict | None:
        try:
            clean = (text or '').strip()

            # Remove markdown code fences like ```json ... ```
            if clean.startswith('```'):
                lines = clean.splitlines()
                if len(lines) >= 3:
                    clean = '\n'.join(lines[1:-1]).strip()

            # If extra text exists, extract first JSON object region
            start = clean.find('{')
            end = clean.rfind('}')
            if start != -1 and end != -1 and end > start:
                clean = clean[start:end + 1]

            data = json.loads(clean)
            if isinstance(data, dict) and 'tool' in data and 'args' in data:
                return data
        except json.JSONDecodeError:
            pass
        return None


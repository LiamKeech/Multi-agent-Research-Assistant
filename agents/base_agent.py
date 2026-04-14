import requests
import json

OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'llama3'

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: list[dict]):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools

    def _chat(self, messages: list[dict]) -> dict:
        """Send messages to Ollama and return the response dict."""
        payload = { 'model': MODEL, 'messages': messages, 'stream': False, }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return response.json()

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
        """Try to parse the model response as a tool call JSON."""
        try:
            # Strip Markdown code fences if present
            clean = text.strip().strip('`').strip()
            if clean.startswith('{'):
                data = json.loads(clean)
                if 'tool' in data and 'args' in data:
                    return data
        except json.JSONDecodeError:
            pass
        return None

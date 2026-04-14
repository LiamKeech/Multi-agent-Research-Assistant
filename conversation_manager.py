import requests
import json
from orchestrator import ResearchOrchestrator

OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'llama3'

# Describe the orchestrator as a tool the LLM can call
RESEARCH_TOOL = {
    'name': 'run_research',
    'description': (
        'Search the web for a topic, summarise the findings, '
        'and return a formatted report with citations. ' 
        'Call this whenever the user asks you to research something.'
    ),
    'params': '{"topic": "string"}'
}

SYSTEM_PROMPT = """You are a helpful research assistant. 
You can answer general questions in plain conversation. 
When the user explicitly asks you to research a topic — for example 
by saying 'research X', 'look up X', or 'find information about X' — 
you MUST respond with ONLY the following JSON and nothing else:
 {"tool": "run_research", "args": {"topic": "<topic>"}} 
 Do NOT use the tool for casual conversation or simple factual questions. 
 Only use it when the user clearly wants a research report."""

class ConversationManager:
    def __init__(self):
        self.orchestrator = ResearchOrchestrator()
        self.history = []

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

    def _parse_tool_call(self, text: str) -> dict | None:
        try:
            clean = text.strip().strip('`').strip()
            if clean.startswith('{'):
                data = json.loads(clean)
                if 'tool' in data and 'args' in data:
                    return data
        except json.JSONDecodeError:
            pass
        return None

    def _build_messages(self, user_input: str) -> list[dict]:
        """Build the full message list: system + history + new user message."""
        system = self.system_message()
        return [system] + self.history + [{'role': 'user', 'content': user_input}]

    def system_message(self) -> dict:
        tool_desc = (
            f'\n\nAvailable tool:\n' 
            f' - {RESEARCH_TOOL["name"]}: {RESEARCH_TOOL["description"]}\n'
            f' Parameters: {RESEARCH_TOOL["params"]}'
        )
        return {'role': 'system', 'content': SYSTEM_PROMPT + tool_desc}

    def send(self, user_input: str) -> str:
        """
        Send a user message, handle any tool call,
        and return the final assistant reply to display to the user.
        """
        messages = self._build_messages(user_input)
        reply = self._chat(messages)

        tool_call = self._parse_tool_call(reply)

        if tool_call and tool_call['tool'] == 'run_research':
            topic = tool_call['args'].get('topic', user_input)
            print(f'\n[ConversationManager] Research triggered for: "{topic}"')

            # Run the full research pipeline
            report = self.orchestrator.run(topic)

            # Inject the report back into the conversation
            # so the model can respond naturally to it
            messages.append({'role': 'assistant', 'content': reply})
            messages.append({'role': 'user',
                             'content': f'Here are the research results:\n\n{report}'})
            final_reply = self._chat(messages)

            # Save the whole exchange to history
            self.history.append({'role': 'user', 'content': user_input})
            self.history.append({'role': 'assistant', 'content': final_reply})
            return final_reply

        else:
            # Plain conversation — no tool needed
            self.history.append({'role': 'user', 'content': user_input})
            self.history.append({'role': 'assistant', 'content': reply})
            return reply
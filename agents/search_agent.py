from agents.base_agent import BaseAgent
from tools.web_search import web_search

TOOLS = [
    {
        'name': 'web_search',
        'description': 'Search the web for information on a topic.',
        'params': '{"query": "string"}'
    }
]

class SearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name='SearchAgent',
            system_prompt=(
                    'You are a search query extraction specialist. '
                    'Your ONLY job is to extract 1 or 2 core keywords from the user prompt and pass them to the web_search tool. '
                    'CRITICAL RULE: DuckDuckGo will fail if you use more than 2 words. '
                    'Example User Input: "Find information about the history of the Apollo 11 moon landing and fact check faked claims." '
                    'Example Tool Call: {"tool": "web_search", "args": {"query": "Apollo 11"}} '
                    'NEVER pass the full sentence. YOU MUST RESPOND WITH ONLY VALID JSON AND NO OTHER TEXT.'
                    + self._build_tool_prompt_static(TOOLS)
            ), tools=TOOLS
        )

    def _build_tool_prompt_static(self, tools):
        # Helper called before super().__init__ completes
        lines = ['\n\nYou have access to the following tools.',
                 'To call a tool, respond ONLY with valid JSON:',
                 '{"tool": "<name>", "args": {<params>}}',
                 'Available tools:']
        for t in tools:
            lines.append(f' - {t["name"]}: {t["description"]}')
        return '\n'.join(lines)

    def run(self, topic: str) -> list[dict]:
        """Search for a topic and return result list."""
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': (
                f'Topic: {topic}\n\n'
                f'Extract ONLY the core 1-2 word noun entity (e.g., "artificial sweeteners", "solid-state battery", "Byzantine Empire") from the topic. '
                f'You MUST respond with ONLY a JSON tool call. Do not add any conversational text.'
            )}
        ]
        reply = self._chat(messages)

        tool_call = self._parse_tool_call(reply)
        if tool_call and tool_call['tool'] == 'web_search':
            query = tool_call['args'].get('query', topic)

            # --- NEW FAILSAFE ---
            # Small LLMs are bad at counting words.
            # Force the query to be a maximum of 2 words in Python.
            words = query.split()
            if len(words) > 2:
                print(f' [SearchAgent] Truncating long query "{query}" to 2 words.')
                query = ' '.join(words[:2])
            # --------------------

            print(f' [SearchAgent] Calling web_search("{query}")')
            return web_search(query)

        # Fallback: model did not use the tool, search directly
        print(f' [SearchAgent] Fallback direct search for: {topic}')

        # Apply the same failsafe to the fallback
        words = topic.split()
        if len(words) > 2:
            topic = ' '.join(words[:2])

        return web_search(topic)

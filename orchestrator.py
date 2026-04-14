import json
import re

import requests

from agents.base_agent import MODEL, OLLAMA_URL
from agents.search_agent import SearchAgent
from agents.summariser_agent import SummariserAgent
from agents.fact_checker_agent import FactCheckerAgent
from agents.citation_agent import CitationAgent

class ResearchOrchestrator:
    AGENT_ORDER = ['SearchAgent', 'SummariserAgent', 'FactCheckerAgent', 'CitationAgent']
    AGENT_DEPENDENCIES = {
        'SummariserAgent': ['SearchAgent'],
        'FactCheckerAgent': ['SearchAgent', 'SummariserAgent'],
        'CitationAgent': ['SearchAgent'],
    }

    def __init__(self):
        print('Initialising agents...')
        self.search_agent = SearchAgent()
        self.summariser_agent = SummariserAgent()
        self.fact_checker_agent = FactCheckerAgent()
        self.citation_agent = CitationAgent()
        self.memory = []
        self.agent_registry = {
            'SearchAgent': self.search_agent,
            'SummariserAgent': self.summariser_agent,
            'FactCheckerAgent': self.fact_checker_agent,
            'CitationAgent': self.citation_agent,
        }
        print('All agents ready.\n')

    def run(self, query: str) -> str:
        print(f'Orchestrator received query: "{query}"\n')
        try:
            print('[Step 1] Dispatching SearchAgent...')
            results = self.search_agent.run(query)
            if not results:
                return 'No results found. Try a different query.'
            print(f' Found {len(results)} results.\n')

            print('[Step 2] Dispatching SummariserAgent...')
            combined_text = ' '.join(r.get('snippet', '') for r in results)
            summary = self.summariser_agent.run(combined_text)
            print(' Summary complete.\n')

            print('[Step 3] Dispatching CitationAgent...')
            citations = self.citation_agent.run(results)
            print(' Citations formatted.\n')

            return self._compile_report(query, summary, citations)

        except Exception as e:
            return (
                f'The research pipeline encountered a problem and could not ' 
                f'complete. Error: {e}. ' 
                f'Check that Ollama is running and try again.'
            )

    def _handle_follow_up(self, query: str) -> str | None:
        """Resolve simple ordinal follow-up questions from session memory."""
        result_index = self._extract_result_index(query)
        if result_index is None:
            return None

        previous_turn = self._latest_search_turn()
        if previous_turn is None:
            return 'I do not have any previous search results in this session yet.'

        results = previous_turn.get('search_results', [])
        if not results:
            return 'I do not have any previous search results in this session yet.'

        if result_index == -1:
            result_index = len(results)

        if result_index < 1 or result_index > len(results):
            return f'I only have {len(results)} result(s) from the most recent search.'

        result = results[result_index - 1]
        title = result.get('title', 'Untitled')
        snippet = result.get('snippet', '').strip()
        url = result.get('url', '').strip()
        source_query = previous_turn.get('query', 'the previous search')

        lines = [
            f'Here is more about result {result_index} from your previous search on "{source_query}":',
            '',
            f'Title: {title}',
        ]
        if snippet:
            lines.extend(['Snippet:', snippet])
        if url:
            lines.extend(['URL:', url])
        return '\n'.join(lines)

    def _extract_result_index(self, query: str) -> int | None:
        """Extract an ordinal result reference like 'third result' from a query."""
        if not query:
            return None

        lowered = query.lower()
        ordinal_words = {
            'first': 1,
            'second': 2,
            'third': 3,
            'fourth': 4,
            'fifth': 5,
            'sixth': 6,
            'seventh': 7,
            'eighth': 8,
            'ninth': 9,
            'tenth': 10,
            'last': -1,
        }

        patterns = [
            r'\b(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|\d+)(?:st|nd|rd|th)?\s+(?:result|item|option)\b',
            r'\b(?:result|item|option)\s+(?P<num>\d+)(?:st|nd|rd|th)?\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            ordinal = match.groupdict().get('ordinal')
            if ordinal:
                if ordinal.isdigit():
                    return int(ordinal)
                return ordinal_words.get(ordinal)
            num = match.groupdict().get('num')
            if num and num.isdigit():
                return int(num)

        return None

    def _latest_search_turn(self) -> dict | None:
        """Return the most recent memory entry that contains search results."""
        for turn in reversed(self.memory):
            if turn.get('search_results'):
                return turn
        return None

    def _plan_agents(self, query: str) -> list[str]:
        """Ask Ollama which agents to use and return a validated agent list."""
        payload = {
            'model': MODEL,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        'You are a routing planner for a research assistant. '
                        'Choose which agents to use for the user query. '
                        'Allowed agents: SearchAgent, SummariserAgent, FactCheckerAgent, CitationAgent. '
                        'Return ONLY a JSON list of agent names. '
                        'Include any prerequisites needed for later agents to work. '
                        'Examples: if you choose SummariserAgent, include SearchAgent first; '
                        'if you choose FactCheckerAgent, include SearchAgent and SummariserAgent first; '
                        'if you choose CitationAgent, include SearchAgent first.'
                    ),
                },
                {
                    'role': 'user',
                    'content': f'Query: {query}\nReturn a JSON list of agent names only.',
                },
            ],
            'stream': False,
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json().get('message', {}).get('content', '')
            parsed = self._parse_agent_list(content)
            if parsed:
                return self._expand_plan(parsed)
        except (requests.RequestException, ValueError):
            pass

        return list(self.AGENT_ORDER)

    def _parse_agent_list(self, text: str) -> list[str]:
        """Parse a JSON list of agent names from the model response."""
        if not text:
            return []

        clean = text.strip()
        if clean.startswith('```'):
            lines = clean.splitlines()
            if len(lines) >= 2:
                clean = '\n'.join(lines[1:-1]).strip()

        start = clean.find('[')
        end = clean.rfind(']')
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end + 1]

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        allowed = set(self.AGENT_ORDER)
        plan = []
        seen = set()
        for item in data:
            if isinstance(item, str) and item in allowed and item not in seen:
                plan.append(item)
                seen.add(item)
        return plan

    def _expand_plan(self, planned_agents: list[str]) -> list[str]:
        """Add required prerequisites and return the plan in canonical execution order."""
        required = set()
        for agent_name in planned_agents:
            self._collect_dependencies(agent_name, required)
            required.add(agent_name)

        return [agent_name for agent_name in self.AGENT_ORDER if agent_name in required]

    def _collect_dependencies(self, agent_name: str, required: set[str]) -> None:
        for dependency in self.AGENT_DEPENDENCIES.get(agent_name, []):
            if dependency not in required:
                self._collect_dependencies(dependency, required)
                required.add(dependency)

    def _compile_report(self, query: str, state: dict) -> str:
        lines = [
            f'RESEARCH REPORT',
            f'Query: {query}',
            f'=' * 50,
            '',
        ]

        if state.get('search_results'):
            lines.extend([
                'SEARCH RESULTS',
                '-' * 30,
            ])
            for i, result in enumerate(state['search_results'], 1):
                title = result.get('title', 'Untitled')
                snippet = result.get('snippet', '').strip()
                url = result.get('url', '').strip()
                lines.append(f'[{i}] {title}')
                if snippet:
                    lines.append(f'    {snippet}')
                if url:
                    lines.append(f'    {url}')
            lines.append('')

        if state.get('summary'):
            lines.extend([
                'SUMMARY',
                '-' * 30,
                state['summary'],
                '',
            ])

        if state.get('fact_check'):
            lines.extend([
                'FACT CHECK',
                '-' * 30,
                state['fact_check'],
                '',
            ])

        if state.get('citations'):
            lines.extend([
                'SOURCES',
                '-' * 30,
            ])
            for i, c in enumerate(state['citations'], 1):
                lines.append(f'[{i}] {c}')

        return '\n'.join(lines)
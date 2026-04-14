from agents.search_agent import SearchAgent
from agents.summariser_agent import SummariserAgent
from agents.citation_agent import CitationAgent

class ResearchOrchestrator:
    def __init__(self):
        print('Initialising agents...')
        self.search_agent = SearchAgent()
        self.summariser_agent = SummariserAgent()
        self.citation_agent = CitationAgent()
        print('All agents ready.\n')

    def run(self, query: str) -> str:
        """
        Orchestrate the full research pipeline:
        1. Search -> 2. Summarise each result -> 3. Cite -> 4. Compile
        """
        print(f'Orchestrator received query: "{query}"\n')

        # ── Step 1: Search ─────────────────────────────────────────
        print('[Step 1] Dispatching SearchAgent...')
        results = self.search_agent.run(query)
        if not results:
            return 'No results found. Try a different query.'
        print(f' Found {len(results)} results.\n')

        # ── Step 2: Summarise ──────────────────────────────────────
        print('[Step 2] Dispatching SummariserAgent...')
        combined_text = ' '.join(r.get('snippet', '') for r in results)
        summary = self.summariser_agent.run(combined_text)
        print(' Summary complete.\n')

        # ── Step 3: Cite ───────────────────────────────────────────
        print('[Step 3] Dispatching CitationAgent...')
        citations = self.citation_agent.run(results)
        print(' Citations formatted.\n')

        # ── Step 4: Compile report ─────────────────────────────────
        report = self._compile_report(query, summary, citations)
        return report

    def _compile_report(self, query, summary, citations):
        lines = [
            f'RESEARCH REPORT',
            f'Query: {query}',
            f'=' * 50,
            '',
            'SUMMARY',
            '-' * 30,
            summary,
            '',
            'SOURCES',
            '-' * 30,
        ]
        for i, c in enumerate(citations, 1):
            lines.append(f'[{i}] {c}')
        return '\n'.join(lines)
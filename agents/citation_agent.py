from agents.base_agent import BaseAgent
from tools.citation import format_citation

class CitationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name='CitationAgent',
            system_prompt=(
                'You are a citation formatting specialist. '
                '' 'Format references accurately in APA style.'
            ),
            tools=[]
        )

    def run(self, results: list[dict]) -> list[str]:
        """Format each search result as a citation string."""
        citations = []
        for r in results:
            citation = format_citation(
                url=r.get('url', ''),
                title=r.get('title', 'Untitled')
            )
            citations.append(citation)
        return citations
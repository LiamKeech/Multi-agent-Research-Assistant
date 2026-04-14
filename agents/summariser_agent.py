from agents.base_agent import BaseAgent, MODEL
from tools.summariser import summarise_text

class SummariserAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name='SummariserAgent',
            system_prompt=(
                'You are a research summarisation expert. '
                'Condense the provided text into clear, concise bullet points.'
            ),
            tools=[]
        )

    def run(self, text: str) -> str:
        """Return a bullet-point summary of text."""
        return summarise_text(text, model=MODEL)

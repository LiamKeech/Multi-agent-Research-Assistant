from agents.base_agent import BaseAgent


class FactCheckerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name='FactCheckerAgent',
            system_prompt=(
                'You are a fact-checking specialist. '
                'Review the provided summary and identify any likely false claims, '
                'unsupported assertions, or statements that should be verified more carefully.'
            ),
            tools=[]
        )

    def run(self, summary: str) -> str:
        """Review a summary for likely false claims."""
        cleaned_summary = (summary or '').strip()
        if not cleaned_summary:
            return 'No summary provided to fact-check.'

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': (
                'Does this summary contain any likely false claims? List any concerns. '
                'If there are no obvious issues, say so clearly.\n\n'
                f'{cleaned_summary}'
            )},
        ]

        reply = self._chat(messages)  # _chat already returns text content
        return (reply or '').strip() or 'No fact-check response was generated.'



AGENTA_PROMPT = """
You are Agent A.

Your job:
- Solve assigned tasks
- Read tool results
- Produce structured output
- Be concise
"""

class AgentA(BaseAgent):
    def __init__(self):
        super().__init__(AGENTA_PROMPT)
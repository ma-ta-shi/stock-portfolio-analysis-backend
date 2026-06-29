from base_agents import BaseAgent

AGENTB_PROMPT = """
You are Manager Agent.

Responsibilities:
- Receive user requests
- Delegate to worker
- Review worker output
- Improve accuracy
- Return final answer
"""

class AgentB(BaseAgent):
    def __init__(self):
        super().__init__(AGENTB_PROMPT)
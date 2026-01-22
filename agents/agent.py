from agno.agent import Agent
from agno.models.groq import Groq
from tools.tool import retrieve_resume_context


def create_resume_agent():
    llm = Groq()  # NO arguments

    agent = Agent(
        name="ResumeQAAgent",
        model=llm,
        tools=[retrieve_resume_context],
        instructions="""
You are a resume question-answering agent.

Rules:
- Always call the tool to retrieve resume content.
- Answer ONLY from retrieved information.
- Do NOT guess.
- If information is missing, say:
  "Not mentioned in the resume."
"""
    )
    return agent

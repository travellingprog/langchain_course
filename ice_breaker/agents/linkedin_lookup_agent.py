from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """given the full name {name_of_person} I want you to get me a link to their
    LinkedIn profile page. Your answer should contain only a URL."""
    prompt_template = PromptTemplate.from_template(template)

    tools = [get_profile_url]

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    linkedin_profile_url = agent_executor.invoke({"input": prompt_template.format(name_of_person=name)})
    return linkedin_profile_url

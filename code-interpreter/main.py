"""
I deviated a lot from the example given in the course.

Instead, I followed the examples provided in the LangChain docs:
- https://python.langchain.com/docs/integrations/toolkits/python
- https://python.langchain.com/docs/integrations/toolkits/csv
"""

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Start...")

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    tools = [PythonREPLTool()]

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    python_agent = create_openai_functions_agent(llm, tools, prompt)
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    #     python_agent_executor.invoke(
    #         {
    #             "input": """Generate and save in the current working directory 15 QR codes
    # that point to www.udemy.com/course/langchain. You have the qrcode package installed already.
    # """
    #         }
    #     )

    csv_agent_executor = create_csv_agent(
        llm=llm,
        path="episode_info.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    csv_agent_executor.invoke(
        {"input": "how many columns are there in file episode_info.csv"}
    )
    csv_agent_executor.invoke(
        {
            "input": "print the seasons in ascending order of the number of episodes they have"
        }
    )


if __name__ == "__main__":
    main()

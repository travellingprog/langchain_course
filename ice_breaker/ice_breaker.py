from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import PersonIntel, person_intel_parser
from third_parties.linkedin import scrape_linkedin_profile


def ice_break(name: str) -> PersonIntel:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(
        gist_url="https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
    )

    summary_prompt_template = PromptTemplate.from_template(
        """
        Given the LinkedIn information {information} about a person I want you to create:
        1. a short summary
        2. two interesting facts about them

        {format_instructions}
    """,
        partial_variables={"format_instructions": person_intel_parser.get_format_instructions()},
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = summary_prompt_template | llm | person_intel_parser

    return chain.invoke({"information": linkedin_data})


if __name__ == "__main__":
    print("hello LangChain!")
    result = ice_break("Eden Marco")
    print(result)

from langchain.tools import tool
from langchain.utilities.serpapi import SerpAPIWrapper


class CustomSerpAPIWrapper(SerpAPIWrapper):
    def __init__(self):
        super(CustomSerpAPIWrapper, self).__init__()

    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from SerpAPI."""
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")

        snippets = []
        for organic_result in res.get("organic_results", []):
            if "link" in organic_result.keys():
                snippets.append(organic_result["link"])

        if len(snippets) > 0:
            return str(snippets)
        else:
            return "No good search result found"


@tool
def get_profile_url(name: str):
    """Useful for when you need to get the LinkedIn profile URL"""
    search = CustomSerpAPIWrapper()
    res = search.run(name)
    return res

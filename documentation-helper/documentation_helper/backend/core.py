from typing import Any

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone

from documentation_helper.consts import EMBEDDING_MODEL, INDEX_NAME


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa.invoke({"query": query})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?")["result"])

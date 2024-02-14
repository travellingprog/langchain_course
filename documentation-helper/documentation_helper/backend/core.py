from typing import Any

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone

from documentation_helper.consts import EMBEDDING_MODEL, INDEX_NAME


def run_llm(query: str, chat_history: list[tuple[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        output_key="result",
    )
    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?")["result"])

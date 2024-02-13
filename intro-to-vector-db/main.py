import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone

load_dotenv()

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader("./mediumblogs/mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    # print(len(texts))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)

    llm = OpenAI()
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retriever = docsearch.as_retriever()
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    result = chain.invoke(
        {"input": "What is a vector DB? Give me a 15 word answer for a beginner"}
    )
    print(result)

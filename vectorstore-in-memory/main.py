from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Hello Vectorstore-in-memory")
    pdf_path = "./ReAct_paper.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=pages)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # how to create a new vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    # how to retrieve an existing vectorstore
    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    res = qa.invoke({"query": "Give me the gist of ReAct in 3 sentences"})
    print(res["result"])

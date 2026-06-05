from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

loader = TextLoader("sample.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

response = qa_chain.invoke(
    {"query": "What is the content of the document?"}
)

print(response["result"])
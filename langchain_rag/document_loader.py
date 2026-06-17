from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template = "What is the main topic of the following text? {text}",
    input_variables = ["text"]
)
parser = StrOutputParser()
loader = TextLoader("crricket_poem.txt", encoding="utf-8")
docs = loader.load()
print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser
print(chain.invoke({"text": docs[0].page_content}))

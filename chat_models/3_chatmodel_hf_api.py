from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id="meta-llama/Llama-3.2-3B",
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    max_new_tokens=256,
    temperature=0.7
)

response = llm.invoke("What is the capital of France?")

print(response)
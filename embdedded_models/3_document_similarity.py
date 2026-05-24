from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

docs = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome."]

query = "What is the capital of France?"

doc_embeddings = embeddings.embed_documents(docs)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

most_similar_doc_index = np.argmax(similarities)
print(f"Most similar document: {docs[most_similar_doc_index]} with similarity score: {similarities[most_similar_doc_index]}")

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, folder_path)

import config

# Start.
llm_client = OpenAI()

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("TEXT_EMBEDDING_MODEL")
)

# # Initialize Chroma client (persists to disk)
# client = chromadb.Client()
# Use PersistentClient (this is the key change)
client = chromadb.PersistentClient(path="./chroma_db")

# TODO: Create a collection in ChromaDB with the name "walmart_knowledge".
# Use client.get_or_create_collection().
# Set the property embedding_function to the embedding function variable defined above.
# Store the result in a variable named "collection".

# ---- Add documents with metadata ----
# TODO: Add 3-4 documents to the ChromaDB collection.
# Set these values in the collection:
#   - documents: the list of documents.
#   - metadatas: A collection of key-value pairs defining the category and region for each document.
#   - ids: A collection of unique IDs defining the category and region for each document.



def retrieve_docs(query, filters=None):
    """Retrieve documents along with distances"""
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where=filters,
        include=["documents", "distances"]
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    return docs, distances

def is_relevant(docs, distances, threshold=0.8):
    """
    Strict relevance check
    Only allow answer if similarity is strong enough
    """
    if not docs:
        return False

    return min(distances) < threshold

def generate_strict_answer(query, docs):
    """Generate answer ONLY from context"""
    context = "\n".join(docs)

    prompt = f"""
# TODO: Define a prompt for the LLM.
# The prompt must:
#   - ask the system to be a strict RAG system.
#   - Define the following rules:
#       - It must only answer using the context that is provided to it.
#       - It must not use any prior knowledge.
#       - If the answer is not present, it must give an appropriate response.
#   - Pass the context and query to the prompt as placeholders.
    """

    response = llm_client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ---- Query ----
query = "How is AI used in supply chain?"
# query = "What is quantum computing?"  # intentionally outside domain

docs, distances = retrieve_docs(query)

print("\nRetrieved Docs:", docs)
print("Distances:", distances)

# ---- Strict RAG Decision ----
if is_relevant(docs, distances):
    print("\nStrict RAG Answer:")
    print(generate_strict_answer(query, docs))
else:
    print("\nStrict RAG Response:")
    print("I don't have enough information in the provided context.")

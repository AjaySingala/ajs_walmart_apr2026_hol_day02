# Full RAG (Grounded Answer).
# Step 3: Full RAG → retrieve + augment + generate grounded answer

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path).
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
from openai import OpenAI

# Initialize OpenAI client using API key
print(f"Initialize client...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load document.
# TODO: Load the file "company_policy.txt" with utf-8 encoding.
# Read the contents in to a variable named "document".


# Chunking.
# TODO: Split the document into chunks based on the new-line character in a variable named "chunks".

# Create embeddings.
print(f"\n Create embeddings...")
embeddings = []
for chunk in chunks:
    if chunk.strip():
        emb = client.embeddings.create(
            model=os.getenv("TEXT_EMBEDDING_MODEL"),
            input=chunk
        )
        # TODO: Append the chunk to the embeddings collection along with the embedding returned by the model.
        # Use data[0].embedding to extract the embedding from the result returned by the model.


# Similarity function.
def similarity(a, b):
    print(f"\n similarity()...")
    return sum(x*y for x, y in zip(a, b))

# Query
query = "What is the meal allowance for employees?"
# query = "What is the internet reimbursement policy?"

# Embed query.
# TODO: Embed the query into a variable named "query_embedding".
# Use data[0].embedding to extract the embedding from the result returned by the model.

# Retrieve top relevant chunk.
print(f"\n Retrieve top relevant chunk...")
best_chunk = max(
    embeddings,
    key=lambda x: similarity(query_embedding, x[1])
)[0]

# Augment prompt with retrieved context.
augmented_prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{best_chunk}

Question:
{query}
"""

# Call LLM with grounded context.
# TODO: Call the LLM using the gpt-40-mini model.
# Pass two messages to the LLM:
#   1. A system role, defining it's role as a precise assistant.
#   2. The user role, passing the augmented prompt define above.
# Store the result in a variable named "response".


print("\n--- RAG Response (Grounded) ---")
print(response.choices[0].message.content)


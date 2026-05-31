import chromadb
from groq import Groq

chroma_client = chromadb.Client()
groq_client = Groq(api_key="")
collection = chroma_client.create_collection("knowledge_base")

with open("knowledge.txt","r") as f:
    text = f.read()

chunks = [line.strip() for line in text.split("\n") if line.strip()
     ]
print("Chunks:", chunks)
print("Total:", len(chunks))

collection.add(
    documents = chunks,
    ids=[str(i) for i in range(len(chunks))]
)

question = input("Ask a question: "
                 )
results = collection.query(
    query_texts=[question],
    n_results=3
)
relevant_chunks = results["documents"][0]

context = "\n".join(relevant_chunks)
prompt = f"""Answer the question using only the context below.
If the answer is not in the context say I don't know.

Context:
{context}

Question: {question}
Answer:"""

response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt}]
)
print("\n AI ANSWER :", response.choices[0].message.content)
from app import get_embedding, index  
from groq import Groq
import os
import json

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY "),
)

def query_documents(question, n_results=4):
   
    query_embedding = get_embedding(question)
   
    # Query the index with the generated embedding
    results = index.query(
        namespace="namespace1",
        vector=query_embedding,
        top_k=n_results,
        include_metadata=True  
    )

    # Extract the relevant chunks
    relevant_chunks = [
        {"id": match['id'], "text": match['metadata']['text']}  # Extract text from metadata
        for match in results['matches']
    ]

    return relevant_chunks



def generate_response(question, relevant_chunks):
    #if not relevant_chunks:
        #return "I have no knowledge about this context."
    
    # Concatenate the text content of the relevant chunks
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    
    # Refined prompt to guide the model's response
    prompt = (
       f"Given the following context, provide a detailed answer to the question asked. "
        f"Only use the information provided in the context. Do not include any additional information or make assumptions.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
        
    )
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-versatile",
    )

    answer = response.choices[0].message.content.strip()


# Ensure the answer is based on the context
    context_keywords = set(context.lower().split())
    answer_keywords = set(answer.lower().split())
    
    if context_keywords.intersection(answer_keywords):
        return answer
    else:
        return "There is no information available in the documents about this topic."""

def main():
    question = input("Ask me any Question: ")
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
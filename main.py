from app import get_embedding, index  
from groq import Groq
import os
import json

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY "),
)

def query_documents(question, n_results=4):
   
    query_embedding = get_embedding(question)
   
    
    results = index.query(
        namespace="namespace1",
        vector=query_embedding,
        top_k=n_results,
        include_metadata=True  
    )

    relevant_chunks = [
        {"id": match['id'], "text": match['metadata']['text']}  
        for match in results['matches']
    ]

    return relevant_chunks



def generate_response(question, relevant_chunks):
    
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    
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
import os
import re
import string
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


load_dotenv()


client = Groq(
    api_key=os.environ.get("GroqGROQ_API_KEY "),
)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

Pine_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "bye"

chunked_documents = []

def clean_text(text):

    text = re.sub(r'<.*?>', '', text)
    
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    cleaned_text = ' '.join(words)
    
    return cleaned_text

def get_embedding(text): 
    response = Pine_client.inference.embed(
        "multilingual-e5-large",
        inputs=text,
        parameters={
            "input_type": "passage"
        }
    )
    embedding = response.data[0].values
    return embedding



index = Pine_client.Index(index_name)

if index_name not in Pine_client.list_indexes().names():
    Pine_client.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )

if __name__ == "__main__":
 def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

 def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

 directory_path = "../news_articles"
 documents = load_documents_from_directory(directory_path)


 for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": clean_text(chunk)})


 for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_embedding(doc["text"])

 for doc in chunked_documents:
    print("==== Inserting chunks into db ====")
    index.upsert(
        vectors=[
            {"id": doc["id"], "values": doc["embedding"], "metadata": {"text": doc["text"]}}
        ],
        namespace="namespace1"
    )
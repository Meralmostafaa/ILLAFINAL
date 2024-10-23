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
    api_key=os.environ.get("GroqGROQ_API_KEY"),
)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

Pine_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "bye"

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

def clean_text(text):
   
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
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

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def add_embeddings_to_pinecone(document_text):

    chunks = split_text(document_text)
    
    for i, chunk in enumerate(chunks):

        cleaned_chunk = clean_text(chunk)
        embedding = get_embedding(cleaned_chunk)
    
        index.upsert(
            vectors=[{
                "id": f"doc_chunk_{i+1}",  
                "values": embedding,
                "metadata": {"text": cleaned_chunk}
            }],
            namespace="namespace1"
        )

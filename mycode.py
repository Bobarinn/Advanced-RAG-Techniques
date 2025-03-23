from openai import OpenAI
from dotenv import load_dotenv
import os
from pypdf import PdfReader

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pdf_reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in pdf_reader.pages]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))


token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

embedding_function = SentenceTransformerEmbeddingFunction()


chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection-4", embedding_function=embedding_function
)


ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)


initial_query = "Unearned revenue by segment: Summarize the table"

def generate_expansion_query(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful expert financial research assistant. 
   Provide more specific queries to the given question up to 5 queries, that might be found in a document like an annual report. The point is to have more queries that will use to query our vector database for better results. seperate by /n"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


expansion_query = generate_expansion_query(initial_query)
print(expansion_query)

expansion_queries = expansion_query.split("\n")

joint_query = f"{initial_query} {expansion_query}"
print(joint_query)



results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)

joint_query_embeddings = embedding_function([joint_query])

results = chroma_collection.query(
    query_embeddings=joint_query_embeddings, n_results=5, include=["documents", "embeddings"]
)


retrieved_documents = results["documents"][0]

for document in retrieved_documents:
    print(document)
    

def final_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message.content
    return answer


final_response(initial_query, retrieved_documents)


import os
import json
import requests
from flask import Flask, request, jsonify
from datetime import datetime
import hashlib
import chromadb  # ChromaDB client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request, jsonify
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma


import uuid  # For generating unique identifiers


# Initialize the Ollama model with the external server URL and specified model
llm = OllamaLLM(
    model="gemma2:27b",
    base_url="http://129.254.169.91:11434"  # External Ollama server address
)

# Instantiate the Flask app
app = Flask(__name__)

# Set the directory for ChromaDB persistent storage
CHROMADB_PERSIST_DIR = "./chromadb_data"  # Directory where ChromaDB SQLite data will be stored
os.makedirs(CHROMADB_PERSIST_DIR, exist_ok=True)  # Ensure directory exists

# Initialize persistent ChromaDB client with SQLite
client = chromadb.Client(Settings(persist_directory=CHROMADB_PERSIST_DIR))
index_name = 'projects'

# Initialize embedding model and text splitter
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a SentenceTransformer model
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Example sizes, adjust as needed

vector_store = Chroma(collection_name=index_name, embedding_function=embedding_model, persist_directory=CHROMADB_PERSIST_DIR)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # Retrieve top 2 relevant documents


# Initialize ChromaDB collection
collection = client.get_or_create_collection(name=index_name)

def is_binary(file_path):
    """Check if a file is binary."""
    with open(file_path, 'rb') as file:
        initial_bytes = file.read(65536)
    return b'\0' in initial_bytes  # Check for null byte which indicates binary file


import concurrent.futures

# Function to split and embed content in batches
def get_chunks_and_embeddings(content):
    # Split text into chunks
    chunks = text_splitter.split_text(content)
    # Batch encode chunks to generate embeddings
    embeddings = embedding_model.encode(chunks, batch_size=32, convert_to_tensor=True)  # Adjust batch size as needed
    return chunks, embeddings

# Function to process a single file and add it to ChromaDB
def process_file(file_path):
    # if is_binary(file_path):
    #     print(f"Skipping binary file: {file_path}")
    #     return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError as e:
        print(f"Skipping file due to encoding error: {file_path}, error: {e}")
        return None

    # Split and embed content
    chunks, embeddings = get_chunks_and_embeddings(content)
    
    # Ensure chunks and embeddings are not empty
    if len(chunks) == 0 or len(embeddings) == 0:
        print(f"No valid chunks or embeddings for file: {file_path}")
        return None
    
    # Insert each chunk as a separate document in ChromaDB
    # file_metadata = {
    #     "file_path": file_path,
    #     "title": os.path.basename(file_path),
    #     "date": datetime.utcnow().isoformat()  # Convert datetime to string
    # }
    documents = []
    embeddings_list = []
    #metadatas = []
    ids = []
    
    for chunk, embedding in zip(chunks, embeddings):
        #%document_id = hashlib.md5((file_path + chunk).encode()).hexdigest()
        document_id = str(uuid.uuid4())
        documents.append(chunk)
        embeddings_list.append(embedding.cpu().numpy().tolist())
        #metadatas.append(file_metadata)
        ids.append(document_id)

    # Check again that the lists are non-empty before adding to the collection    
    #if documents and embeddings_list and metadatas and ids:
    if documents and embeddings_list and ids:
        # Batch add to ChromaDB
        collection.add(
            documents=documents,
            embeddings=embeddings_list,
            #metadatas=metadatas,
            ids=ids
        )
        #print(f"Successfully added file to ChromaDB: {file_path}")
    else:
        print(f"Skipping addition to ChromaDB due to empty lists for file: {file_path}")

    return file_path


# Optimized function to load data with embeddings into ChromaDB
def create_and_load_index():
    if collection.count() > 0:
        print(f"Collection '{index_name}' already populated. Loading existing data.")
    else:
        print("Loading new data into ChromaDB...")
        
        # Use concurrent futures for parallel file processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process files in parallel
            futures = []
            for root, _, files in os.walk('projects'):
                for file in files:
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(process_file, file_path))
            
            # Collect results to track progress
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                # if result:
                #     print(f"Processed: {result}")
        
        print("Data loaded into ChromaDB.")

    # Check if files are present in the directory
    print("Files in CHROMADB_PERSIST_DIR after loading:")
    for root, dirs, files in os.walk(CHROMADB_PERSIST_DIR):
        for file in files:
            print(os.path.join(root, file))


# # Function to split and embed content
# def get_chunks_and_embeddings(content):
#     # Split text into chunks
#     chunks = text_splitter.split_text(content)
#     # Generate embeddings for each chunk using SentenceTransformer
#     embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
#     return chunks, embeddings

# # Function to load data with embeddings into ChromaDB
# def create_and_load_index():
#     # Check if the collection is empty to avoid reloading data
#     if collection.count() > 0:
#         print(f"Collection '{index_name}' already populated. Loading existing data.")
#     else:
#         print("Loading new data into ChromaDB...")
#         for root, _, files in os.walk('projects'):
#             for file in files:
#                 file_path = os.path.join(root, file)

#                 # Skip binary files
#                 if is_binary(file_path):
#                     print(f"Skipping binary file: {file_path}")
#                     continue
                
#                 try:
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         content = f.read()
#                 except UnicodeDecodeError as e:
#                     print(f"Skipping file due to encoding error: {file_path}, error: {e}")
#                     continue

#                 # Split and embed content
#                 chunks, embeddings = get_chunks_and_embeddings(content)
                
#                 # Insert each chunk as a separate document in ChromaDB
#                 for chunk, embedding in zip(chunks, embeddings):
#                     document_id = hashlib.md5((file_path + chunk).encode()).hexdigest()
#                     collection.add(
#                         documents=[chunk],
#                         embeddings=[embedding.cpu().numpy().tolist()],  # Convert tensor to list for storage
#                         metadatas=[{
#                             "file_path": file_path,
#                             "title": os.path.basename(file),
#                             "date": datetime.utcnow().isoformat()  # Convert datetime to string
#                         }],
#                         ids=[document_id]
#                     )
#         print("Data loaded into ChromaDB.")

#     # Check if files are present in the directory
#     print("Files in CHROMADB_PERSIST_DIR after loading:")
#     for root, dirs, files in os.walk(CHROMADB_PERSIST_DIR):
#         for file in files:
#             print(os.path.join(root, file))


# Function to search documents using vector similarity
def search_documents(query, top_k=2):
    # Retrieve relevant documents using the retriever
    retrieved_documents = retriever.get_relevant_documents(query)

    documents = []
    seen_hashes = set()
    for doc in retrieved_documents:
        content = doc.page_content
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            documents.append(content)
            if len(documents) == top_k:
                break

    for i, content in enumerate(documents):
        print(f"Document {i+1}:\nContent: {content}\n")

    return documents


# # Function to search documents using vector similarity
# def search_documents(query, top_k=5):
#     query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().tolist()
#     if query_embedding is None:
#         return []

#     # Perform similarity search in ChromaDB
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k,
#         include=['documents']
#         #include=['documents', 'metadatas']
#     )
    
#     documents = []
#     seen_hashes = set()
#     for result in results['documents'][0]:
#         content = result
#         content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
#         if content_hash not in seen_hashes:
#             seen_hashes.add(content_hash)
#             #metadata = results['metadatas'][0][results['documents'][0].index(result)]
#             #file_path = metadata.get("file_path")
#             #title = metadata.get("title")
#             #documents.append((file_path, title, content))
#             documents.append(content)
#             if len(documents) == top_k:
#                 break

#     for i, content in enumerate(documents):
#         print(f"Document {i+1}:\nContent: {content}\n")

#     # for i, (file_path, title, content) in enumerate(documents):
#     #     print(f"Document {i+1}:\nPath: {file_path}\nTitle: {title}\nContent: {content}\n")
    
#     return documents


# Function to generate answer using LLM server
def generate_answer(documents, query):
    context = "\n".join([doc[2] for doc in documents[:2]])

    # prompt_template = PromptTemplate.from_template(
    #     "You are a knowledgeable coding assistant. Based on the following context, provide an "
    #     "answer to the question, including exact keyword matching code skeleton where appropriate. "
    #     "Ensure that the code is clear, well-commented, and tailored to the specific requirements "
    #     "of the question. \n\nContext:\n{context}\n\nQuestion: {question}"
    # )
    
    prompt = "You are a knowledgeable coding assistant. Based on the following context, provide an "
    prompt += "answer to the question, including exact keyword matching code skeleton where appropriate. "
    prompt += "Ensure that the code is clear, well-commented, and tailored to the specific requirements "
    prompt += "of the question. \n\nContext:\n"
    prompt += context
    prompt += "\n\nQuestion: "
    prompt += query

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use "stuff" or another chain type if needed
        retriever=retriever)


    # Generate the answer using the query
    try:
        result = qa_chain.invoke({'query': prompt})
        answer = result["result"]
        source_documents = result.get("source_documents", [])
    except Exception as e:
        print(f"An error occurred while generating the answer: {e}")
        answer = "An error occurred while generating the answer. Please try again later."
        source_documents = []

    # Print source documents if needed
    for i, doc in enumerate(source_documents):
        print(f"Source Document {i+1}: {doc.metadata['title']}\n{doc.page_content[:200]}...\n")

    return answer
    
    # prompt_template = ChatPromptTemplate.from_template(
    #     "You are a knowledgeable coding assistant. Based on the following context, provide a "
    #     "answer to the question, including exact keyword matching code skeleton where appropriate. "
    #     "Ensure that the code is clear, well-commented, and tailored to the specific requirements "
    #     "of the question. \n\nContext:\n{context}\n\nQuestion: {question}"
    # )

    # Create the LangChain LLM chain with the prompt and external Ollama model
    #chain = LLMChain(llm=llm, prompt=prompt_template)
    

    # answer = ''

    # try:
    #     #answer = chain.run({"context": context, "question": query})
    #     answer = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=client.as_)
    # except requests.RequestException as e:
    #     print(f"An error occurred while generating the answer: {e}")
    #     answer = "An error occurred while generating the answer. Please try again later."

    # return answer

# Endpoint to handle RAG queries
@app.route('/query', methods=['POST'])
def query_rag():
    query = request.get_json().get('query')

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    documents = search_documents(query)

    if not documents:
        return jsonify({"error": "No relevant documents found for the query."}), 404

    answer = generate_answer(documents, query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    create_and_load_index()
    app.run(host='0.0.0.0', port=5000)

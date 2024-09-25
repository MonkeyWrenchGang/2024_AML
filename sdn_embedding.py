"""
File: sdn_embeddings.py
Description: This script creates a ChromaDB collection of embeddings from the OFAC SDN list. 
             It uses a SentenceTransformer model to generate embeddings for names and stores 
             them in ChromaDB. It also provides a function to query similar names from the 
             collection based on cosine similarity.
Author: Mike A.
Date: 24 Sept 2024
"""

import warnings
import re
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Constants
COLLECTION_NAME = "sdn_names"

def clean_name(name):
    """
    Clean a name by lowercasing and removing unnecessary punctuation.
    
    Args:
        name (str): The name to be cleaned.
    
    Returns:
        str: Cleaned name string.
    
    Example:
        clean_name("Name-to  Be Cleane,d") -> "name-to be cleaned"
    """
    name = name.lower()
    name = re.sub(r'[^a-zA-Z\s\-\'\.]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def query_similar_names(query_name, n_results=3):
    """
    Queries the ChromaDB collection to find the top N similar names.
    
    Args:
        query_name (str): The name to search for similar entries.
        n_results (int): The number of similar entries to return (default: 3).
    
    Returns:
        list of tuples: Each tuple contains (similar_name, distance).
    """
    cleaned_query_name = clean_name(query_name)
    query_embedding = model.encode([cleaned_query_name])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    similar_names = results["documents"][0]
    distances = results["distances"][0]
    
    return list(zip(similar_names, distances))

# Initialize ChromaDB client and create or retrieve the collection
client = chromadb.PersistentClient(path="./vect")

try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' has been deleted.")
except Exception as e:
    print(f"Collection '{COLLECTION_NAME}' could not be deleted: {str(e)}")

collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
print(f"Collection '{COLLECTION_NAME}' has been created.")

# Load the SentenceTransformer model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the CSV file into a pandas DataFrame
csv_file_path = "/Users/mikeames/env_sdn/data/SDN_individuals.csv"
df = pd.read_csv(csv_file_path)

# Clean the 'sdn_name' column and prepare embeddings
df['sdn_name'] = df['sdn_name'].astype(str).apply(clean_name)
sdn_names = df['sdn_name'].tolist()
ent_nums = df['ent_num'].tolist()

embeddings = model.encode(sdn_names)

# Add embeddings to the ChromaDB collection
for i, name in enumerate(sdn_names):
    doc_id = str(ent_nums[i])  # Convert 'ent_num' to a string
    collection.add(
        documents=[name],
        embeddings=[embeddings[i].tolist()],
        ids=[doc_id],  # Use the string-converted 'ent_num' as the ID
        metadatas=[{"name": name}]
    )

print("Embeddings added to ChromaDB.")

# Example query
query_name = "AL ZAWAHIRI, Ayman"
n_results = 5
similar_names_with_distances = query_similar_names(query_name, n_results)

# Display the results
print(f"Top {n_results} similar names to '{query_name}':")
for name, distance in similar_names_with_distances:
    similarity = 1 - distance  # Convert distance back to cosine similarity
    print(f"Name: {name}, Distance: {distance:,.4f}, Cosine Similarity: {similarity:,.4f}")

# Display collection information
print(f"Collection '{COLLECTION_NAME}' contains {collection.count()} names.")

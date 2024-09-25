"""
File: sdn_similarity_search.py
Description: This script loads a dataset, performs similarity searches using embeddings stored in ChromaDB,
             and appends the top 5 matched names, distances, and cosine similarities to the dataset.
             It also outputs a summary where ent_num equals match_1_id, providing the count and mean similarity.
Author: Mike A.
Date: 24 sept 2024
"""

import pandas as pd
import re
import chromadb
from sentence_transformers import SentenceTransformer
import warnings

# Suppress Hugging Face FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Function to clean names
def clean_name(name):
    """
    Cleans a name by converting to lowercase and removing unnecessary punctuation.

    Args:
        name (str): The name to clean.

    Returns:
        str: Cleaned name.
    """
    if not isinstance(name, str):
        return ""
    
    name = name.lower()
    name = re.sub(r'[^a-zA-Z\s\-\'\.]', '', name)  # Retain hyphens and apostrophes
    name = re.sub(r'\s+', ' ', name)  # Normalize multiple spaces
    return name.strip()

# Function to query ChromaDB for the top N similar names, including document ID and cosine similarity
def query_similar_names(query_name, n_results=5):
    """
    Queries the ChromaDB collection to find the top N similar names based on cosine similarity.

    Args:
        query_name (str): The name to search for.
        n_results (int): The number of similar names to return.

    Returns:
        list of tuples: Each tuple contains (document_id, similar_name, distance, cosine_similarity).
    """
    cleaned_query_name = clean_name(query_name)
    query_embedding = model.encode([cleaned_query_name])[0]

    # Perform a similarity search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    similar_names = results["documents"][0]
    distances = results["distances"][0]
    document_ids = results["ids"][0]

    # Return document ID, name, distance, and similarity as a tuple
    return [(document_ids[i], similar_names[i], distances[i], 1 - distances[i]) for i in range(n_results)]

# Step 1: Connect to the ChromaDB collection
persist_directory = "./vect"  # Path where your ChromaDB collection is stored
client = chromadb.PersistentClient(path=persist_directory)

# Load the existing collection
collection_name = "sdn_names"
collection = client.get_collection(name=collection_name)

print(f"Connected to collection '{collection_name}'.")

# Step 2: Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Load the new dataset with names
new_csv_file_path = "/Users/mikeames/env_sdn/data/alt.csv"
df_new = pd.read_csv(new_csv_file_path).sample(2000)  # Load sample data for testing

# Ensure 'alt_name' column is free of NaN and convert to string (avoid inplace=True)
df_new['alt_name'] = df_new['alt_name'].fillna('')

# Step 4: Add columns for the top 5 matches, distances, and similarities
for i in range(1, 6):
    df_new[f'match_{i}_id'] = ''
    df_new[f'match_{i}_name'] = ''
    df_new[f'match_{i}_distance'] = 0.0
    df_new[f'match_{i}_similarity'] = 0.0

# Step 5: Run similarity tests for each name in the dataset
for index, row in df_new.iterrows():
    query_name = row['alt_name']
    top_matches = query_similar_names(query_name, n_results=5)

    # Append the top 5 matches, document IDs, distances, and similarities to the DataFrame
    for i, (doc_id, match_name, distance, similarity) in enumerate(top_matches):
        df_new.at[index, f'match_{i+1}_id'] = doc_id
        df_new.at[index, f'match_{i+1}_name'] = match_name
        df_new.at[index, f'match_{i+1}_distance'] = distance
        df_new.at[index, f'match_{i+1}_similarity'] = similarity

# Step 6: Save the new dataset with similarity results to a CSV file
output_csv_file_path = "/Users/mikeames/env_sdn/data/output_dataset.csv"
df_new.to_csv(output_csv_file_path, index=False)

# Step 7: Generate summary where ent_num == match_1_id
if 'ent_num' in df_new.columns:
    # Ensure 'ent_num' is a string for comparison
    df_new['ent_num'] = df_new['ent_num'].astype(str)
    
    # Add a column to check if ent_num equals match_1_id
    df_new['is_exact_match'] = df_new['ent_num'] == df_new['match_1_id']
    
    # Filter rows where ent_num equals match_1_id
    exact_matches = df_new[df_new['is_exact_match']]
    
    # Count the number of exact matches
    exact_match_count = exact_matches.shape[0]
    
    # Calculate the mean similarity for exact matches
    mean_similarity = exact_matches['match_1_similarity'].mean()
    
    print(f"Summary of exact matches:")
    print(f"Count of exact matches: {exact_match_count} out of {df_new.shape[0]}")
    print(f"Mean similarity for exact matches: {mean_similarity:.4f}")
else:
    print("The column 'ent_num' does not exist in the dataset.")

print(f"Similarity results saved to {output_csv_file_path}")

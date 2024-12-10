# import os
# import pandas as pd
# import numpy as np
# from langchain_openai.embeddings import OpenAIEmbeddings  
# import faiss
# from dotenv import load_dotenv

# load_dotenv()

# # Set your OpenAI API key
# openai_api_key = os.getenv('OPENAI_API_KEY')
# if not openai_api_key:
#     openai_api_key = input("Enter your OpenAI API key: ")

# # Read the CSV files into pandas DataFrames
# run_data = pd.read_csv('run_data.csv')  # Replace with your run_data.csv file path
# test_data = pd.read_csv('test_data.csv')  # Replace with your test_data.csv file path

# # Fill NaN values with empty strings
# test_data['Natural Language Query'] = test_data['Natural Language Query'].fillna('')
# test_data['Alternatives'] = test_data['Alternatives'].fillna('')
# test_data['Steps'] = test_data['Steps'].fillna('')

# run_data['Natural Language Query'] = run_data['Natural Language Query'].fillna('')

# # Extract texts from the test_data columns
# test_texts_col1 = test_data['Natural Language Query'].tolist()
# test_texts_col2 = test_data['Alternatives'].tolist()
# test_texts_col3 = test_data['Steps'].tolist()

# # Initialize the OpenAIEmbeddings object from LangChain
# embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key, model='text-embedding-3-small')

# # Function to generate embeddings using LangChain
# def generate_embeddings(texts):
#     embeddings = embeddings_model.embed_documents(texts)
#     embeddings = np.array(embeddings).astype('float32')
#     # Normalize embeddings
#     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#     embeddings = embeddings / norms
#     return embeddings

# # Generate and normalize embeddings for test_data
# print("Generating embeddings for 'Natural Language Query' in test_data...")
# test_embeddings_col1 = generate_embeddings(test_texts_col1)

# print("Generating embeddings for 'Alternatives' in test_data...")
# test_embeddings_col2 = generate_embeddings(test_texts_col2)

# print("Generating embeddings for 'Steps' in test_data...")
# test_embeddings_col3 = generate_embeddings(test_texts_col3)

# # Get the embedding dimension
# embedding_dim = test_embeddings_col1.shape[1]

# # Create FAISS indices for test_data embeddings
# index_col1 = faiss.IndexFlatIP(embedding_dim)
# index_col2 = faiss.IndexFlatIP(embedding_dim)
# index_col3 = faiss.IndexFlatIP(embedding_dim)

# # Add test_data embeddings to the indices
# index_col1.add(test_embeddings_col1)
# index_col2.add(test_embeddings_col2)
# index_col3.add(test_embeddings_col3)

# # Perform similarity search for each natural language query in run_data
# run_texts = run_data['Natural Language Query'].tolist()

# results = []
# print("Performing similarity search for run_data queries...")
# for query in run_texts:
#     # Embed the query
#     query_embedding = embeddings_model.embed_query(query)
#     query_embedding = np.array(query_embedding).astype('float32')
#     query_embedding = query_embedding / np.linalg.norm(query_embedding)

#     # Perform similarity search in each index
#     D_col1, I_col1 = index_col1.search(np.array([query_embedding]), 3)
#     D_col2, I_col2 = index_col2.search(np.array([query_embedding]), 3)
#     D_col3, I_col3 = index_col3.search(np.array([query_embedding]), 3)

#     # Initialize arrays to hold similarities
#     num_rows = len(test_data)
#     similarities_col1 = np.zeros(num_rows)
#     similarities_col2 = np.zeros(num_rows)
#     similarities_col3 = np.zeros(num_rows)

#     # Map similarities back to the test_data rows
#     for idx, sim in zip(I_col1[0], D_col1[0]):
#         similarities_col1[idx] = sim

#     for idx, sim in zip(I_col2[0], D_col2[0]):
#         similarities_col2[idx] = sim

#     for idx, sim in zip(I_col3[0], D_col3[0]):
#         similarities_col3[idx] = sim

#     # Compute the average similarity per row
#     average_similarities = (similarities_col1 + similarities_col2 + similarities_col3) / 3

#     # Get the indices of the top 3 matches
#     top_k_indices = np.argsort(average_similarities)[-3:][::-1]

#     # Retrieve information for the top 3 matches
#     top_matches = []
#     for idx in top_k_indices:
#         row = test_data.iloc[idx]
#         top_matches.append({
#             'SQL Query': row['SQL Query'],
#             'Type': row['Type'],
#             'Natural Language Query': row['Natural Language Query'],
#             'Alternatives': row['Alternatives'],
#             'Steps': row['Steps'],
#             'Filtered Schema': row.get('Filtered Schema', '')  
#         })
    
#     # Store the results
#     results.append({
#         'Run Query': query,
#         'Top Matches': top_matches
#     })

# # Output results
# for result in results:
#     print(f"\nRun Query: {result['Run Query']}")
#     print("Top Matches:")
#     for i, match in enumerate(result['Top Matches'], start=1):
#         print(f"\nMatch {i}:")
#         for key, value in match.items():
#             print(f"{key}: {value}")



import os
import pandas as pd
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings  
import faiss
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    openai_api_key = input("Enter your OpenAI API key: ")

# Parameter to set the number of matches (k)
num_matches = 3  # Change this value as needed

# Read the CSV files into pandas DataFrames
run_data = pd.read_csv('run_data_1000.csv')  # Replace with your run_data.csv file path
test_data = pd.read_csv('test_data_1000.csv')  # Replace with your test_data.csv file path

# Fill NaN values with empty strings
test_data['Natural Language Query'] = test_data['Natural Language Query'].fillna('')
test_data['Alternatives'] = test_data['Alternatives'].fillna('')
test_data['Steps'] = test_data['Steps'].fillna('')

run_data['Natural Language Query'] = run_data['Natural Language Query'].fillna('')
run_data['list_of_tables'] = run_data['list_of_tables'].apply(eval)  # Convert strings to lists
test_data['list_of_tables'] = test_data['list_of_tables'].apply(eval)  # Convert strings to lists

# Extract texts from the test_data columns
test_texts_col1 = test_data['Natural Language Query'].tolist()
test_texts_col2 = test_data['Alternatives'].tolist()
test_texts_col3 = test_data['Steps'].tolist()

# Initialize the OpenAIEmbeddings object from LangChain
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key, model='text-embedding-3-small')

# Function to generate embeddings using LangChain
def generate_embeddings(texts):
    embeddings = embeddings_model.embed_documents(texts)
    embeddings = np.array(embeddings).astype('float32')
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings

# Generate and normalize embeddings for test_data
print("Generating embeddings for 'Natural Language Query' in test_data...")
test_embeddings_col1 = generate_embeddings(test_texts_col1)

print("Generating embeddings for 'Alternatives' in test_data...")
test_embeddings_col2 = generate_embeddings(test_texts_col2)

print("Generating embeddings for 'Steps' in test_data...")
test_embeddings_col3 = generate_embeddings(test_texts_col3)

# Get the embedding dimension
embedding_dim = test_embeddings_col1.shape[1]

# Create FAISS indices for test_data embeddings
index_col1 = faiss.IndexFlatIP(embedding_dim)
index_col2 = faiss.IndexFlatIP(embedding_dim)
index_col3 = faiss.IndexFlatIP(embedding_dim)

# Add test_data embeddings to the indices
index_col1.add(test_embeddings_col1)
index_col2.add(test_embeddings_col2)
index_col3.add(test_embeddings_col3)

# Perform similarity search and list_of_tables comparison for each query in run_data
results = []
complete_matches = 0
partial_matches = 0
partial_miss_average = 0
no_matches = 0
fractional_results = []

print("Performing similarity search and list_of_tables comparison...")
for query, list1, unq_alias_run in zip(
        run_data['Natural Language Query'],
        run_data['list_of_tables'],
        run_data['UNQ_ALIAS']):
    # Embed the query
    query_embedding = embeddings_model.embed_query(query)
    query_embedding = np.array(query_embedding).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Perform similarity search in each index
    D_col1, I_col1 = index_col1.search(np.array([query_embedding]), num_matches)
    D_col2, I_col2 = index_col2.search(np.array([query_embedding]), num_matches)
    D_col3, I_col3 = index_col3.search(np.array([query_embedding]), num_matches)

    # Retrieve matching rows
    top_matches_indices = np.unique(np.concatenate([I_col1[0], I_col2[0], I_col3[0]]))
    list2 = set()
    test_nl_questions = []
    unq_aliases_test = []
    for idx in top_matches_indices:
        list2.update(test_data.iloc[idx]['list_of_tables'])
        test_nl_questions.append(test_data.iloc[idx]['Natural Language Query'])
        unq_aliases_test.append(test_data.iloc[idx]['UNQ_ALIAS'])

    # Compare list1 with list2
    list2 = set(list2)
    matching_tables = [item for item in list1 if item in list2]
    if set(list1).issubset(list2):
        result = 0
        complete_matches += 1
    elif any(item in list2 for item in list1):
        matching_fraction = len(matching_tables) / len(list1)
        result = 1 - matching_fraction
        fractional_results.append(matching_fraction)
        partial_matches += 1
    else:
        result = -1
        no_matches += 1

    # Store the result
    results.append({
        'UNQ_ALIAS_run': unq_alias_run,
        'run_nl_question': query,
        'list_of_tables_run': list1,
        'UNQ_ALIAS_test': unq_aliases_test,
        'test_nl_questions': test_nl_questions,
        'list_of_tables_test': list(list2),
        'Result': result,
        'Matching Tables': matching_tables
    })

# Calculate summary metrics
partial_miss_average = sum(fractional_results) / len(fractional_results) if fractional_results else 0

# Create a new DataFrame for results
results_df = pd.DataFrame(results)

# Add summary rows to a separate DataFrame
summary_data = {
    'Result Totals': ['total complete matching', 'total partial matching', 'partial misses (average)', 'total 0 matches'],
    'No of Matching Tables': [complete_matches, partial_matches, partial_miss_average, no_matches]
}
summary_df = pd.DataFrame(summary_data)

# Save the results and summary to a new CSV file
results_df.to_csv('run_test_results_test_1000.csv', index=False)
summary_df.to_csv('run_test_summary_test_1000.csv', index=False)

# Print summary metrics
print(f"\nTotal complete matching: {complete_matches}")
print(f"Total partial matching: {partial_matches}")
print(f"Partial misses (average): {partial_miss_average}")
print(f"Total 0 matches: {no_matches}")

print("\nResults saved to 'run_test_results_test_1000.csv'.")
print("Summary saved to 'run_test_summary_test_1000.csv'.")
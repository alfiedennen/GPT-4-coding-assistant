import os
import json
import re
import numpy as np
import fnmatch
import torch
from annoy import AnnoyIndex
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.decomposition import PCA

# Load the OpenAI Codex tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Create a code completion pipeline using the Codebert model
code_completion = pipeline('text-generation', model=model, tokenizer=tokenizer)

def preprocess_code(code, file_type):
    # Remove single-line comments
    if file_type in ['py', 'js']:
        code = re.sub(r'//.*', '', code)

    # Remove multi-line comments
    if file_type in ['py', 'js', 'css']:
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)

    # Remove HTML comments
    if file_type == 'html':
        code = re.sub(r'<!--[\s\S]*?-->', '', code)

    # Normalize whitespace
    code = re.sub(r'\n\s*\n', '\n\n', code)
    code = re.sub(r'\t', '    ', code)

    # Additional preprocessing for better context
    code = re.sub(r'\n{3,}', '\n\n', code)  # Remove excessive blank lines
    code = re.sub(r'[^\x00-\x7F]+', '', code)  # Remove non-ASCII characters

    return code

def generate_embeddings_for_code(code):
    max_len = 510
    code_chunks = [code[i:i + max_len] for i in range(0, len(code), max_len)]

    chunk_embeddings = []
    for chunk in code_chunks:
        if len(chunk) > 0:
            inputs = tokenizer.encode_plus(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state

            chunk_embedding = embeddings.mean(dim=1).squeeze().tolist()
            chunk_embeddings.append(chunk_embedding)

    if len(chunk_embeddings) > 0:
        chunk_embeddings_array = np.vstack(chunk_embeddings)
        code_embedding = chunk_embeddings_array.mean(axis=0).tolist()

        # Apply PCA to reduce the dimensionality of the embeddings
        reduced_code_embedding = reduce_embedding_dimension([code_embedding])

        return reduced_code_embedding[0]  # Return the first (and only) item in the list
    else:
        return []
    
def generate_embeddings_for_file(file_path):
    with open(file_path, 'r') as f:
        code = f.read()

    file_type = file_path.split('.')[-1]
    processed_code = preprocess_code(code, file_type)
    embeddings = generate_embeddings_for_code(processed_code)

    return embeddings

def generate_embeddings_for_all_files(website_path):
    embeddings_dict = {}
    annoy_index = AnnoyIndex(768, 'angular')  # 768 is the embedding dimension
    index_map = {}

    index_counter = 0
    for root, _, files in os.walk(website_path):
        for ext in ('*.py', '*.html', '*.css', '*.js'):
            for file in fnmatch.filter(files, ext):
                file_path = os.path.join(root, file)
                embeddings = generate_embeddings_for_file(file_path)
                embeddings_dict[file_path] = embeddings

                annoy_index.add_item(index_counter, embeddings)
                index_map[index_counter] = file_path
                index_counter += 1

                print(f"Vectorized file: {file_path}")

    annoy_index.build(50)  # 50 trees for fast approximate search
    annoy_index.save('embeddings.ann')
    with open('embeddings.json', 'w') as f:
        json.dump(embeddings_dict, f)
    with open('index_map.json', 'w') as f:
        json.dump(index_map, f)


def reduce_embedding_dimension(embeddings, n_components=200):
    if len(embeddings) > 1:
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
    else:
        reduced_embeddings = embeddings
    return reduced_embeddings

if __name__ == "__main__":
    generate_embeddings_for_all_files("code")
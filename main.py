import openai
import requests
import json
import sys
from create_embeddings import generate_embeddings_for_code, preprocess_code
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from annoy import AnnoyIndex

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def calculate_token_count(conversation_history):
    token_count = 0
    for message in conversation_history:
        token_count += len(tokenizer.tokenize(message["content"], truncation=True, max_length=8000))
    return token_count

def find_most_similar_files(annoy_index, index_map, user_input_embeddings, top_n=10):
    try:
        most_similar_indices = annoy_index.get_nns_by_vector(user_input_embeddings, top_n)
        most_similar_files = [index_map[str(idx)] for idx in most_similar_indices]  # Convert idx to str
        print(f"Most similar files: {most_similar_files}")
        return most_similar_files
    except KeyError as e:
        print(f"KeyError: {e}")
        return []

def get_api_key_from_file(filename):
    with open(filename, 'r') as f:
        api_key = f.read().strip()
    return api_key

openai.api_key = get_api_key_from_file('openaikey.txt')


def interact_with_gpt(app, annoy_index, index_map, user_input, conversation_history, embeddings_lookup_enabled):
    context = ""
    most_similar_files = [] 
    if embeddings_lookup_enabled:
        app.logger.info(f"Searching in embeddings for user input: {user_input}")
        processed_user_input = preprocess_code(user_input, 'py')
        user_input_embeddings = generate_embeddings_for_code(processed_user_input)
        most_similar_files = find_most_similar_files(annoy_index, index_map, user_input_embeddings)

        for file_path in most_similar_files:
            with open(file_path, 'r') as f:
                code_snippet = f.read()[:500]  # Limit the length of the code snippet
                context += f"\n\n--- {file_path} ---\n{code_snippet}\n"
        relevant_code_files = ', '.join([f"{i+1}. {file}" for i, file in enumerate(most_similar_files)])
        conversation_history.append({"role": "user", "content": f"Relevant code files: {relevant_code_files}"})
    else:
        app.logger.info(f"Embeddings lookup disabled for user input: {user_input}")

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": context})

    token_usage = calculate_token_count(conversation_history)
    token_limit = 8000

    if token_usage >= token_limit:
        app.logger.warning("Token limit reached. Clearing conversation history and asking user to rephrase the query.")
        conversation_history.clear()
        conversation_history.append({"role": "system", "content": "You are a GPT-4 coding assistant helping with code."})
        assistant_response = "Warning: Conversation history cleared due to reaching the token limit. Please rephrase your query."
        conversation_history.append({"role": "assistant", "content": assistant_response})
        return {"response": assistant_response, "token_usage": 0}

    data = {
        "model": "gpt-4",
        "messages": conversation_history,
        "max_tokens": 1200,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    print("Sending request to OpenAI API...")
    app.logger.info("Sending request to OpenAI API...")
    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    print("Received response from OpenAI API")
    app.logger.info("Received response from OpenAI")
                    
    response_data = response.json()

    print("Conversation history:", conversation_history)
    print("Response data:", response_data)

    if response_data.get("choices") and response_data["choices"][0].get("message"):
        assistant_response = response_data["choices"][0]["message"]["content"].strip()
        conversation_history.append({"role": "assistant", "content": assistant_response})

        # Add the content of the most similar files to the response dictionary
        most_similar_file_contents = []
        for file_path in most_similar_files:
            with open(file_path, 'r') as f:
                file_content = f.read()
                most_similar_file_contents.append(file_content)

                print("Assistant response:", assistant_response)  # Debug print
                result = {"response": assistant_response, "token_usage": token_usage, "most_similar_files": most_similar_files, "most_similar_file_contents": most_similar_file_contents}
                return result

        result = {"response": assistant_response, "token_usage": token_usage, "most_similar_files": most_similar_files, "most_similar_file_contents": most_similar_file_contents}
        return result     
               
with open('embeddings.json', 'r') as f:
    embeddings_dict = json.load(f)

with open('index_map.json', 'r') as f:
    index_map = json.load(f)

annoy_index = AnnoyIndex(768, 'angular')  # 768 is the embedding dimension
annoy_index.load('embeddings.ann')

print(f"Annoy index loaded: {annoy_index.get_n_items()} items")

conversation_history = [{"role": "system", "content": "You are a GPT-4 coding assistant helping with code."}]
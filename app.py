from flask import Flask, render_template, request, jsonify
import json
from main import interact_with_gpt, conversation_history, annoy_index, index_map
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# Set up logging
app.logger.setLevel(logging.INFO)
app.logger.addHandler(logging.StreamHandler())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user_message', methods=['POST'])
def handle_user_message():
    data = request.json
    app.logger.info(f"Received user_message event: {data}")
    user_input = data.get("message")
    embeddings_lookup_enabled = data.get("embeddings_lookup", "false") == "true"

    if user_input:
        result = interact_with_gpt(app, annoy_index, index_map, user_input, conversation_history, embeddings_lookup_enabled)

        if embeddings_lookup_enabled and "most_similar_files" in result:
            relevant_code_files = '\n\n'.join([f"{i+1}. {file}" for i, file in enumerate(result["most_similar_files"])])
            result["relevant_code_files"] = relevant_code_files
            result["embeddings_referenced"] = relevant_code_files

        return jsonify(result)  # Return the response as JSON
    
if __name__ == '__main__':
    app.run(debug=True)
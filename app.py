from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the BERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load precomputed legal data
try:
    legal_data = pd.read_pickle('./legal_data_with_embeddings.pkl')
    if 'embedding' not in legal_data.columns:
        raise ValueError("The .pkl file does not contain 'embedding' column")
except FileNotFoundError:
    print("Error: The .pkl file was not found.")
    legal_data = None
except Exception as e:
    print(f"Error loading .pkl file: {e}")
    legal_data = None

# Function to compute embeddings
def compute_embeddings(text_list):
    tokens = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if legal_data is None:
        return jsonify({"error": "Legal data not properly loaded"}), 500

    user_input = request.json.get('scenario', '')  # Get user scenario from JSON
    if not user_input:
        return jsonify({"error": "No scenario provided"}), 400

    # Compute embedding for the user's input
    user_embedding = compute_embeddings([user_input]).cpu().numpy()

    # Retrieve embeddings from legal data
    try:
        scenario_embeddings = torch.stack([torch.tensor(embed) for embed in legal_data['embedding']]).cpu().numpy()
    except KeyError:
        return jsonify({"error": "Embedding column missing in legal data"}), 500

    # Compute cosine similarity
    similarities = cosine_similarity(user_embedding, scenario_embeddings)
    best_match_index = similarities.argmax()

    # Retrieve best match information
    best_section = legal_data.iloc[best_match_index]['Section']
    best_description = legal_data.iloc[best_match_index]['Description']
    best_score = similarities[0][best_match_index]

    # Return response
    return jsonify({
        "section": best_section,
        "description": best_description,
        "similarity": float(best_score)
    })

if __name__ == '__main__':
    app.run(debug=True)

import json
import uuid
import os
from flask import Flask, request, jsonify, render_template
from mistralai import Mistral

app = Flask(__name__)

# Get API key from environment variable
client = Mistral(api_key=os.environ.get("moO0jReu6yP6B7tsDYYMXfSpiGbIp8S3"))

# Load documents
with open("docs.json") as f:
    documents = json.load(f)

chunks = []
embeddings = []

def chunk_text(text, size=400):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

def create_embedding(text):
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding

# Prepare chunks and embeddings
for doc in documents:
    for chunk in chunk_text(doc["content"]):
        chunks.append(chunk)
        embeddings.append(create_embedding(chunk))

sessions = {}

def similarity(a, b):
    return sum(x*y for x, y in zip(a, b))

def retrieve(query):

    query_embedding = create_embedding(query)

    sims = []
    for emb in embeddings:
        sims.append(similarity(query_embedding, emb))

    top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]

    results = []
    scores = []

    for i in top_idx:
        results.append(chunks[i])
        scores.append(float(sims[i]))

    return results, scores


def generate_answer(context, history, question):

    prompt = f"""
You are a support assistant.

Answer ONLY using the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Conversation:
{history}

User question:
{question}
"""

    response = client.chat.complete(
        model="mistral-small-latest",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.json

    if "message" not in data:
        return jsonify({"error": "message required"}), 400

    session_id = data.get("sessionId", str(uuid.uuid4()))
    message = data["message"]

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id][-5:]

    retrieved, scores = retrieve(message)

    if not scores or max(scores) < 0.5:
        reply = "I don't have enough information to answer that."
        tokens = 0
    else:

        context = "\n".join(retrieved)

        reply = generate_answer(context, history, message)

        tokens = len(reply.split())

    sessions[session_id].append({
        "user": message,
        "assistant": reply
    })

    return jsonify({
        "reply": reply,
        "tokensUsed": tokens,
        "retrievedChunks": len(retrieved),
        "sessionId": session_id
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

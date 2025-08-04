from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Setup Flask app
app = Flask(__name__)

# ✅ CORS: Allow GitHub Pages frontend
CORS(app, resources={r"/*": {"origins": "https://lxriva.github.io"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://lxriva.github.io"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

# Load FAISS index
embedding = OpenAIEmbeddings()
faiss_db = FAISS.load_local("comicvine_index", embedding, allow_dangerous_deserialization=True)
retriever = faiss_db.as_retriever()

# Create prompt
prompt_template = PromptTemplate.from_template(
    """
You are a helpful and enthusiastic Comics librarian. You love all comics and love giving deeply thought-out responses.
Be descriptive, and make sure to switch up the kind of language you use from response to response, but keep the structure of responses uniform.
If the user asks about comic appearances or interactions of characters, give more in-depth answers that are precise and informative, mention the relevant issues and central storyline as well.
Answer the user's comic-related question using the following context:

{context}

User: {question}
Librarian:
"""
)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("question")
    if not query:
        return jsonify({"error": "Missing question"}), 400

    try:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)  # ✅ Initialize inside request
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template}
        )
        answer = qa_chain.run(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

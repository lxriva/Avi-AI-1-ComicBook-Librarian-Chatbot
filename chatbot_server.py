from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Setup
import os

llm = ChatOpenAI(model_name="gpt-4", temperature=0)


from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app)  # 🔥 Enables CORS for all routes

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

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("question")
    if not query:
        return jsonify({"error": "Missing question"}), 400

    try:
        answer = qa_chain.run(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys
import traceback

# Load environment variables
load_dotenv()

# Setup Flask app
app = Flask(__name__)

# CORS configuration
CORS(app, 
     origins=["https://lxriva.github.io", "http://localhost:3000", "http://127.0.0.1:3000"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# Global variables for lazy loading
faiss_db = None
retriever = None
embedding = None

def initialize_ai_components():
    """Initialize AI components with error handling"""
    global faiss_db, retriever, embedding
    
    if faiss_db is not None:
        return True
    
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        
        print("Initializing OpenAI embeddings...")
        embedding = OpenAIEmbeddings()
        
        print("Loading FAISS index...")
        faiss_db = FAISS.load_local("comicvine_index", embedding, allow_dangerous_deserialization=True)
        retriever = faiss_db.as_retriever()
        
        print("✅ AI components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing AI components: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

# Handle preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "https://lxriva.github.io")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS")
        return response

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "Comic Librarian is running!",
        "version": "1.0",
        "timestamp": str(os.popen('date').read().strip())
    })

@app.route("/debug", methods=["GET"])
def debug():
    # Check if AI components can be initialized
    ai_status = initialize_ai_components()
    
    return jsonify({
        "status": "Railway server is running",
        "environment": {
            "has_openai_key": bool(os.environ.get("OPENAI_API_KEY")),
            "openai_key_length": len(os.environ.get("OPENAI_API_KEY", "")) if os.environ.get("OPENAI_API_KEY") else 0,
            "port": os.environ.get("PORT", "Not set"),
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "files_in_directory": os.listdir('.') if os.path.exists('.') else "Cannot read directory"
        },
        "ai_components": {
            "faiss_initialized": faiss_db is not None,
            "retriever_ready": retriever is not None,
            "embedding_ready": embedding is not None,
            "initialization_status": ai_status
        }
    })

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Check request data
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
            
        data = request.json
        query = data.get("question")
        
        if not query:
            return jsonify({"error": "Missing question field"}), 400
        
        # Check OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            return jsonify({"error": "OpenAI API key not configured"}), 500
        
        # Initialize AI components if not already done
        if not initialize_ai_components():
            return jsonify({
                "error": "AI components failed to initialize. This might be due to missing FAISS index files."
            }), 500
        
        # Import here to avoid issues if packages aren't available
        from langchain_openai import ChatOpenAI
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        # Create prompt template
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
        
        print(f"Processing question: {query}")
        
        # Initialize LLM
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        answer = qa_chain.run(query)
        print(f"Generated answer: {answer[:100]}...")
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(error_msg)
        print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "error": error_msg,
            "type": type(e).__name__
        }), 500

# Standard Railway port handling
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 for Railway
    print(f"Starting server on port {port}")
    print(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Use gunicorn in production, Flask dev server locally
    if os.environ.get("RAILWAY_ENVIRONMENT"):
        # Production on Railway
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        # Local development
        app.run(host="0.0.0.0", port=port, debug=True)

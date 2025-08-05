from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys
import traceback

# Load environment variables (only needed locally)
load_dotenv()

# Setup Flask app
app = Flask(__name__)

# ✅ COMPREHENSIVE CORS SETUP
CORS(app, 
     resources={r"/*": {"origins": "*"}}, 
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=False)

# ✅ EXPLICIT OPTIONS HANDLER for all routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        return response

# ✅ ENSURE CORS headers on every response
@app.after_request
def ensure_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Global variables for lazy loading
faiss_db = None
retriever = None
embedding = None

def initialize_ai_components():
    global faiss_db, retriever, embedding
    
    if faiss_db is not None:
        return True
    
    try:
        print("🔄 Starting AI component initialization...")
        
        # Check if required files exist
        if not os.path.exists("comicvine_index"):
            print("❌ FAISS index directory 'comicvine_index' not found")
            return False
            
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        
        print("🔄 Initializing OpenAI embeddings...")
        embedding = OpenAIEmbeddings()
        
        print("🔄 Loading FAISS index...")
        faiss_db = FAISS.load_local("comicvine_index", embedding, allow_dangerous_deserialization=True)
        retriever = faiss_db.as_retriever()
        
        print("✅ AI components initialized successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing required package: {str(e)}")
        print("Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error initializing AI components: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

@app.route("/", methods=["GET"])
def health_check():
    try:
        return jsonify({
            "status": "Comic Librarian is running!",
            "version": "1.0",
            "environment": "Railway" if os.environ.get("RAILWAY_ENVIRONMENT") else "Local",
            "timestamp": str(os.popen('date').read().strip()) if os.name != 'nt' else "Windows"
        })
    except Exception as e:
        return jsonify({
            "status": "Comic Librarian is running!",
            "version": "1.0",
            "error": str(e)
        })

@app.route("/debug", methods=["GET"])
def debug():
    try:
        ai_status = initialize_ai_components()
        
        # Safe file listing
        try:
            files = os.listdir('.')
        except:
            files = ["Cannot read directory"]
            
        return jsonify({
            "status": "Railway server is running",
            "environment": {
                "has_openai_key": bool(os.environ.get("OPENAI_API_KEY")),
                "openai_key_length": len(os.environ.get("OPENAI_API_KEY", "")) if os.environ.get("OPENAI_API_KEY") else 0,
                "port": os.environ.get("PORT", "Not set"),
                "railway_env": os.environ.get("RAILWAY_ENVIRONMENT", "Not set"),
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "files_in_directory": files,
                "faiss_index_exists": os.path.exists("comicvine_index")
            },
            "ai_components": {
                "faiss_initialized": faiss_db is not None,
                "retriever_ready": retriever is not None,
                "embedding_ready": embedding is not None,
                "initialization_status": ai_status
            }
        })
    except Exception as e:
        return jsonify({
            "error": f"Debug endpoint error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    # Handle preflight requests
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        return response
    
    try:
        print(f"📥 Received request: {request.method}")
        print(f"📥 Content-Type: {request.content_type}")
        print(f"📥 Headers: {dict(request.headers)}")
        
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
            
        data = request.json
        query = data.get("question")
        
        print(f"📝 Question received: {query}")
        
        if not query or query.strip() == "":
            return jsonify({"error": "Missing or empty question field"}), 400
        
        if not os.environ.get("OPENAI_API_KEY"):
            return jsonify({"error": "OpenAI API key not configured"}), 500
        
        print("🔄 Initializing AI components...")
        if not initialize_ai_components():
            return jsonify({
                "error": "AI components failed to initialize. This might be due to missing FAISS index files or missing dependencies."
            }), 500
        
        print("🔄 Setting up LangChain components...")
        from langchain_openai import ChatOpenAI
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
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
        
        print(f"🤖 Processing question: {query}")
        
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        print("🔄 Generating answer...")
        answer = qa_chain.run(query)
        print(f"✅ Generated answer: {answer[:100]}...")
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "error": error_msg,
            "type": type(e).__name__,
            "traceback": traceback.format_exc() if os.environ.get("RAILWAY_ENVIRONMENT") != "production" else None
        }), 500

# ✅ Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ✅ Proper Railway port handling
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug_mode = not bool(os.environ.get("RAILWAY_ENVIRONMENT"))
    
    print(f"🚀 Starting server on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"🌍 Environment: {'Railway' if os.environ.get('RAILWAY_ENVIRONMENT') else 'Local'}")
    
    # For Railway, we need to bind to 0.0.0.0
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)

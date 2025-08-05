from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)

# CORS for all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
faiss_db = None
retriever = None
embedding = None

def initialize_ai_components():
    global faiss_db, retriever, embedding

    if faiss_db is not None:
        return True

    try:
        print("🔄 Starting AI component initialization...")

        if not os.path.exists("comicvine_index"):
            print("❌ FAISS index directory 'comicvine_index' not found")
            return False

        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        print("🔄 Initializing OpenAI embeddings...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment variables")
            return False

        print(f"🔐 API Key: {api_key[:6]}...{api_key[-4:]} length={len(api_key)}")

        # Set API key properly for OpenAIEmbeddings
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
            print("✅ OpenAIEmbeddings initialized successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize OpenAIEmbeddings: {str(e)}")
            traceback.print_exc()
            return False


        print("🔄 Loading FAISS index...")
        faiss_db = FAISS.load_local("comicvine_index", embedding, allow_dangerous_deserialization=True)
        retriever = faiss_db.as_retriever()

        print("✅ AI components initialized successfully!")
        return True

    except ImportError as e:
        print(f"❌ Missing package: {str(e)}")
        return False
    except Exception as e:
        print("❌ Error initializing AI components:")
        traceback.print_exc()
        return False

@app.route("/", methods=["GET"])
def health_check():
    try:
        env = os.environ.get("RAILWAY_ENVIRONMENT")
        return jsonify({
            "status": "Comic Librarian is running!",
            "version": "1.0",
            "environment": env if env else "Cloud Run",
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
        files = os.listdir('.') if os.path.exists('.') else ["Cannot read directory"]
        return jsonify({
            "status": "Server is running",
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
            "error": f"Debug error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        return response

    try:
        data = request.get_json()
        query = data.get("question", "").strip()

        if not query:
            return jsonify({"error": "Missing or empty question field"}), 400
        if not os.environ.get("OPENAI_API_KEY"):
            return jsonify({"error": "OpenAI API key not configured"}), 500

        print(f"📝 Question: {query}")
        if not initialize_ai_components():
            return jsonify({"error": "AI components failed to initialize."}), 500

        from langchain_openai import ChatOpenAI
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        prompt_template = PromptTemplate.from_template("""
You are a helpful Comics librarian. Use the following context:
{context}
User: {question}
Librarian:
""")

        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template}
        )

        try:
            print("⚙️ Running qa_chain...")
            answer = qa_chain.run(query)
            print(f"✅ Answer: {answer}")
            return jsonify({"answer": answer})
        except Exception as e:
            print(f"❌ Error in qa_chain: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "error": f"LLM failed to process: {str(e)}",
                "traceback": traceback.format_exc()
            }), 500


    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route("/favicon.ico")
def favicon():
    return "", 204

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug_mode = not bool(os.environ.get("RAILWAY_ENVIRONMENT"))
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)

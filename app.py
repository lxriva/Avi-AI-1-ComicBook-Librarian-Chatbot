from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys
import traceback
import logging
from pydantic.v1 import ValidationError

# Setup logging to ensure logs appear in Cloud Run
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

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

    logger.debug("\U0001F7E2 initialize_ai_components() called.")

    if faiss_db is not None:
        logger.debug("\u2705 FAISS already initialized.")
        return True

    try:
        logger.debug("\uD83D\uDD04 Starting AI component initialization...")

        if not os.path.exists("comicvine_index"):
            logger.debug("\u274C FAISS index directory 'comicvine_index' not found")
            return False

        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        logger.debug("\U0001FAAA Step A: Preparing to initialize OpenAI embeddings...")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.debug("\u274C OPENAI_API_KEY not found.")
            return False

        logger.debug(f"\U0001F512 API Key present, length={len(api_key)}")

        try:
            embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)
            logger.debug("\u2705 OpenAIEmbeddings initialized.")
        except ValidationError as ve:
            logger.debug(f"\u274C ValidationError during embedding init: {ve}")
            traceback.print_exc()
            return False
        except Exception as e:
            logger.debug(f"\u274C General exception during embedding init: {e}")
            traceback.print_exc()
            return False

        logger.debug("\uD83D\uDD04 Loading FAISS index...")
        faiss_db = FAISS.load_local("comicvine_index", embedding, allow_dangerous_deserialization=True)
        retriever = faiss_db.as_retriever()

        logger.debug("\u2705 FAISS index and retriever initialized.")
        return True

    except Exception as e:
        logger.debug(f"\u274C General error in initialize_ai_components: {e}")
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
        logger.debug("\U0001F50D /debug endpoint accessed.")
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

        logger.debug(f"\U0001F4DD Question: {query}")
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
            logger.debug("\u2699\uFE0F Running qa_chain...")
            answer = qa_chain.run(query)
            logger.debug(f"\u2705 Answer: {answer}")
            return jsonify({"answer": answer})
        except Exception as e:
            logger.debug(f"\u274C Error in qa_chain: {str(e)}")
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

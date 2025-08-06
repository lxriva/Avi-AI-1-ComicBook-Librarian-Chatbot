from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys
import traceback
import logging
from pydantic import ValidationError
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS   


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

    logger.debug("🟢 initialize_ai_components() called.")

    if faiss_db is not None:
        logger.debug("✅ FAISS already initialized.")
        return True

    try:
        logger.debug("🔄 Starting AI component initialization...")

        if not os.path.exists("comicvine_index"):
            logger.debug("❌ FAISS index directory 'comicvine_index' not found")
            return False

        # Log FAISS version
        logger.debug(f"🔎 FAISS version: {faiss.__version__}")

        logger.debug("🪪 Step A: Preparing to initialize OpenAI embeddings...")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.debug("❌ OPENAI_API_KEY not found.")
            return False

        logger.debug(f"🔐 API Key present, length={len(api_key)}")

        try:
            # Try the simplest initialization first
            logger.debug("🔄 Trying basic OpenAI initialization...")
            embedding = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
            logger.debug("✅ OpenAIEmbeddings initialized with basic method.")
            
        except Exception as basic_e:
            logger.debug(f"❌ Basic method failed: {basic_e}")
            
            try:
                # Try with explicit API key
                logger.debug("🔄 Trying with explicit API key...")
                embedding = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=api_key
                )
                logger.debug("✅ OpenAIEmbeddings initialized with explicit key.")
                
            except Exception as explicit_e:
                logger.debug(f"❌ Explicit key method failed: {explicit_e}")
                
                try:
                    # Try with older model
                    logger.debug("🔄 Trying with older model ada-002...")
                    embedding = OpenAIEmbeddings(
                        model="text-embedding-ada-002"
                    )
                    logger.debug("✅ OpenAIEmbeddings initialized with ada-002.")
                    
                except Exception as ada_e:
                    logger.debug(f"❌ All embedding initialization methods failed.")
                    logger.debug(f"Basic error: {basic_e}")
                    logger.debug(f"Explicit error: {explicit_e}")
                    logger.debug(f"Ada error: {ada_e}")
                    return False

        logger.debug("🔄 Loading FAISS index...")
        try:
            faiss_db = FAISS.load_local(
                "comicvine_index", 
                embedding, 
                allow_dangerous_deserialization=True
            )
            retriever = faiss_db.as_retriever()
            logger.debug("✅ FAISS index and retriever initialized.")
        except Exception as faiss_e:
            logger.debug(f"❌ FAISS loading error: {faiss_e}")
            traceback.print_exc()
            return False

        return True

    except Exception as e:
        logger.debug(f"❌ General error in initialize_ai_components: {e}")
        traceback.print_exc()
        return False


@app.route("/", methods=["GET"])
def health_check():
    try:
        env = os.environ.get("RAILWAY_ENVIRONMENT", "Cloud Run")
        return jsonify({
            "status": "Comic Librarian is running!",
            "version": "2.0",
            "environment": env,
            "api_key_present": bool(os.environ.get("OPENAI_API_KEY"))
        })
    except Exception as e:
        return jsonify({
            "status": "Comic Librarian is running!",
            "version": "2.0",
            "error": str(e)
        })


@app.route("/debug", methods=["GET"])
def debug():
    try:
        logger.debug("🔍 /debug endpoint accessed.")
        
        # Get version info
        langchain_version = "Unknown"
        openai_version = "Unknown"
        pydantic_version = "Unknown"
        
        try:
            import langchain
            langchain_version = langchain.__version__
        except:
            pass
            
        try:
            import openai
            openai_version = openai.__version__
        except:
            pass
            
        try:
            import pydantic
            pydantic_version = pydantic.__version__
        except:
            pass
        
        # Try to initialize AI components (but don't fail if it doesn't work)
        ai_status = False
        try:
            ai_status = initialize_ai_components()
        except Exception as e:
            logger.debug(f"Debug: AI initialization failed with: {e}")
        
        files = os.listdir('.') if os.path.exists('.') else ["Cannot read directory"]
        
        return jsonify({
            "status": "Server is running",
            "versions": {
                "langchain": langchain_version,
                "openai": openai_version,
                "pydantic": pydantic_version,
                "python": sys.version
            },
            "environment": {
                "has_openai_key": bool(os.environ.get("OPENAI_API_KEY")),
                "openai_key_length": len(os.environ.get("OPENAI_API_KEY", "")) if os.environ.get("OPENAI_API_KEY") else 0,
                "port": os.environ.get("PORT", "Not set"),
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


@app.route("/test-embedding", methods=["GET"])
def test_embedding():
    """Test embedding initialization without FAISS"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return jsonify({"error": "No API key found"}), 400
        
        # Try to create a simple embedding
        test_embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Try to embed a simple text
        result = test_embedding.embed_query("test")
        
        return jsonify({
            "status": "success",
            "embedding_length": len(result),
            "sample": result[:5]  # First 5 dimensions
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Embedding test failed: {str(e)}",
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

        logger.debug(f"📝 Question: {query}")
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

        try:
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template}
            )

            logger.debug("⚙️ Running qa_chain...")
            answer = qa_chain.run(query)
            logger.debug(f"✅ Answer: {answer}")
            return jsonify({"answer": answer})
        except Exception as e:
            logger.debug(f"❌ Error in qa_chain: {str(e)}")
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
    
    # Use a more conservative gunicorn configuration
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)

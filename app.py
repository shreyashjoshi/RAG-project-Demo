"""Flask backend (app.py) that serves the frontend and provides RAG endpoints.

This is a renamed and slightly extended version of the previous backend.py.
It enables CORS for local testing and serves `frontend/index.html` at '/'.
"""

import os
import shutil
import uuid
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import OpenAI, PromptTemplate, LLMChain
from dotenv import load_dotenv
import openai
import logging
import traceback
import importlib
import importlib.metadata


# Load .env if present
load_dotenv()
# Configure OpenAI client explicitly (langchain wrappers also read the env var).
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Configure basic logging to file for debugging embedding build failures
logging.basicConfig(level=logging.INFO, filename="app.log", filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s")

# Configuration
UPLOAD_FOLDER = "datasets"
ALLOWED_EXTENSIONS = {"txt", "md"}
CHROMA_DIRNAME = "chroma"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # allow cross-origin requests during local testing
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload, create a dataset id and build Chroma embeddings.

    Returns JSON: {"dataset_id": "..."}
    """
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "file type not allowed"}), 400

    filename = secure_filename(file.filename)
    dataset_id = str(uuid.uuid4())
    dataset_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset_id)
    os.makedirs(dataset_path, exist_ok=True)

    file_path = os.path.join(dataset_path, filename)
    file.save(file_path)

    # Ensure OpenAI key is present before attempting to build embeddings
    if not os.environ.get("OPENAI_API_KEY"):
        # Clean up the saved file
        shutil.rmtree(dataset_path, ignore_errors=True)
        logging.error("OPENAI_API_KEY not set; cannot build embeddings for %s", dataset_id)
        return (
            jsonify({
                "error": "failed to build embeddings",
                "details": "OPENAI_API_KEY not set in environment"
            }),
            500,
        )

    # Build Chroma DB for this dataset
    try:
        build_chroma_for_dataset(dataset_id, file_path)
    except Exception as e:
        # Clean up on failure
        shutil.rmtree(dataset_path, ignore_errors=True)
        tb = traceback.format_exc()
        logging.error("Failed to build embeddings for %s: %s\n%s", dataset_id, str(e), tb)
        # Return a concise error to the client and log the full traceback on the server
        return (
            jsonify({"error": "failed to build embeddings", "details": str(e)}),
            500,
        )

    return jsonify({"dataset_id": dataset_id})


def build_chroma_for_dataset(dataset_id: str, file_path: str):
    """Load the uploaded file, split text, embed and persist to disk under the
    dataset directory.
    """
    dataset_path = os.path.join(UPLOAD_FOLDER, dataset_id)
    chroma_path = os.path.join(dataset_path, CHROMA_DIRNAME)

    # Remove existing chroma dir if present (clean build)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    # Load the file into a Document using TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_documents(docs)

    # Create embeddings and persist
    try:
        embeddings = OpenAIEmbeddings()
    except TypeError as e:
        # This commonly happens when the installed langchain-openai/langchain
        # versions are incompatible and an unexpected keyword is being passed
        # to the pydantic model (for example 'proxies'). Log and raise a
        # clearer error message to help debugging.
        # Attempt to capture installed package versions to aid debugging
        try:
            lc_ver = importlib.metadata.version("langchain")
        except Exception:
            lc_ver = "(not installed)"
        try:
            lco_ver = importlib.metadata.version("langchain-openai")
        except Exception:
            lco_ver = "(not installed)"
        logging.error("OpenAIEmbeddings instantiation failed: %s", str(e))
        logging.error("langchain version=%s langchain-openai version=%s", lc_ver, lco_ver)
        raise RuntimeError(
            "Failed to create OpenAIEmbeddings. This often indicates incompatible package versions "
            "(langchain, langchain-openai). Ensure you installed the project's pinned requirements. "
            "See requirements.txt and try: pip install -r requirements.txt"
        ) from e
    db = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)
    db.persist()


@app.route("/ask", methods=["POST"])
def ask():
    """Answer a question against a dataset.

    Expected JSON body: {"dataset_id": "...", "question": "..."}
    """
    body = request.get_json(force=True)
    dataset_id = body.get("dataset_id")
    question = body.get("question")
    if not dataset_id or not question:
        return jsonify({"error": "dataset_id and question required"}), 400

    dataset_path = os.path.join(UPLOAD_FOLDER, dataset_id)
    chroma_path = os.path.join(dataset_path, CHROMA_DIRNAME)
    if not os.path.exists(chroma_path):
        return jsonify({"error": "dataset not found"}), 404

    # Load Chroma DB (uses the persisted directory)
    try:
        embeddings = OpenAIEmbeddings()
    except TypeError as e:
        try:
            lc_ver = importlib.metadata.version("langchain")
        except Exception:
            lc_ver = "(not installed)"
        try:
            lco_ver = importlib.metadata.version("langchain-openai")
        except Exception:
            lco_ver = "(not installed)"
        logging.error("OpenAIEmbeddings instantiation failed in /ask: %s", str(e))
        logging.error("langchain version=%s langchain-openai version=%s", lc_ver, lco_ver)
        return (
            jsonify({
                "error": "failed to create embeddings",
                "details": "OpenAIEmbeddings() failed. Possible incompatible package versions; see server logs."
            }),
            500,
        )
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    # Simple retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    # Build a concise prompt that includes the retrieved snippets
    snippets = "\n\n---\n\n".join([d.page_content for d in docs])
    prompt_template = """
You are an assistant that answers questions using only the provided context below.
If the answer is not contained in the context, say "I don't know." Keep answers short.

Context:
{context}

Question: {question}
Answer:
"""

    prompt = prompt_template.format(context=snippets, question=question)

    # Call OpenAI for the final answer using langchain's OpenAI wrapper
    llm = OpenAI(temperature=0)
    # Use a very small chain for simplicity
    chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["context", "question"], template=prompt_template))
    answer = chain.run({"context": snippets, "question": question})

    return jsonify({"answer": answer.strip()})


@app.route('/datasets/<dataset_id>/<path:filename>', methods=['GET'])
def serve_dataset_file(dataset_id, filename):
    """Optional helper to fetch uploaded files for debugging."""
    dataset_path = os.path.join(UPLOAD_FOLDER, dataset_id)
    return send_from_directory(dataset_path, filename)


@app.route('/health', methods=['GET'])
def health():
    """Return a small JSON status for basic diagnostics."""
    return jsonify({
        "ok": True,
        "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
    })


@app.route('/')
def serve_frontend_index():
    """Serve the static frontend index for convenience (same-origin)."""
    return send_from_directory('frontend', 'index.html')


if __name__ == "__main__":
    # For local testing only. In Kubernetes you'll run via a proper WSGI server.
    app.run(host="0.0.0.0", port=8080, debug=True)

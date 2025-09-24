"""Minimal Flask backend for RAG-based QA over uploaded files.

This backend exposes two endpoints:
  - POST /upload : accepts a file upload (markdown or text), builds a Chroma DB
                    in a temporary directory and returns a dataset id.
  - POST /ask    : accepts JSON {"dataset_id": "...", "question": "..."}
                    performs retrieval against the Chroma DB and returns
                    an answer generated with OpenAI.

Notes:
  - This implementation is intentionally small and synchronous. For
    production you'd want background jobs, authentication, rate limits,
    and better error handling.
  - The OpenAI API key must be set in the OPENAI_API_KEY environment variable.
  - The script uses a simple per-upload dataset directory under ./datasets/
    identified by a generated dataset id.
"""

import os
import shutil
import uuid
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import OpenAI, PromptTemplate, LLMChain
from dotenv import load_dotenv


# Load .env if present
load_dotenv()

# Configuration
UPLOAD_FOLDER = "datasets"
ALLOWED_EXTENSIONS = {"txt", "md"}
CHROMA_DIRNAME = "chroma"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
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

    # Build Chroma DB for this dataset
    try:
        build_chroma_for_dataset(dataset_id, file_path)
    except Exception as e:
        # Clean up on failure
        shutil.rmtree(dataset_path, ignore_errors=True)
        return jsonify({"error": "failed to build embeddings", "details": str(e)}), 500

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
    embeddings = OpenAIEmbeddings()
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
    embeddings = OpenAIEmbeddings()
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


if __name__ == "__main__":
    # For local testing only. In Kubernetes you'll run via a proper WSGI server.
    app.run(host="0.0.0.0", port=8000, debug=True)

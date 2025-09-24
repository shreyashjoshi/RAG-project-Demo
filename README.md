RAG Demo (minimal)
===================

This repository demonstrates a minimal two-tier RAG (Retrieval-Augmented
Generation) application with:

- Frontend: static HTML/JS for uploading a text/markdown file and chatting with it.
- Backend: Flask app that builds a Chroma DB per upload and answers questions using OpenAI.
- Kubernetes manifests: example Deployment/Service YAML for backend and frontend.

Important: This is an educational minimal example. Do not use as-is in production.

Prerequisites
-------------

- Python 3.11+
- An OpenAI API key set in the environment as OPENAI_API_KEY
- kubectl and access to a Kubernetes cluster if you want to deploy

Install dependencies (local testing)
----------------------------------

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Run locally
-----------

Start the backend (for development):

```powershell
python backend.py
```

Open `frontend/index.html` in your browser. For local testing, the frontend expects the backend at the same origin; if you open the file directly the browser will attempt to call relative paths on the file:// origin — instead, run a simple static server or update the `API_BASE` in `index.html` to point at `http://localhost:8000`.

Simple static server (PowerShell):

```powershell
python -m http.server 8080 --directory frontend
```

Now open http://localhost:8080 and use the UI. Upload a small text or markdown file and ask questions.

Deploy to Kubernetes (example)
-----------------------------

These manifests are intentionally simple and use hostPath mounts for demonstration only. Replace the images and volumes with proper container images and persistent volumes for production.

1. Edit `k8s/backend-deployment.yaml` and replace the OPENAI_API_KEY value with a Kubernetes Secret or inject it via your CI system.
2. Copy the local project files into the nodes or build container images that contain `backend.py` and `frontend/` static files.
3. Apply manifests:

```powershell
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/frontend-service.yaml
```

Notes and next steps
--------------------

- The backend currently rebuilds embeddings per upload and stores them under `datasets/<dataset_id>/chroma`.
- To avoid re-embedding everything you can implement incremental updates, file watchers, or background workers.
- Add authentication, rate-limiting, and input validation before exposing this service publicly.

License
-------
This code is provided under MIT-style terms — adapt as you need.
# Langchain RAG Demo — Detailed README

This small repo demonstrates a Retrieval-Augmented Generation (RAG) pipeline
using LangChain wrappers, OpenAI embeddings, and Chroma as a local vector
store. It includes both an offline indexing pipeline and a lightweight FastAPI
backend that supports uploading a single document and asking questions about it.

Overview of the components
--------------------------
- `create_database.py` — Offline script that loads markdown files from
  `data/books/`, splits them into overlapping chunks, embeds them using
  `OpenAIEmbeddings`, and persists a Chroma vector store to `chroma/`.
- `query_data.py` — Simple CLI that loads the persisted `chroma/` store,
  retrieves top-k chunks for a query, and calls `ChatOpenAI` with a prompt
  built from the retrieved context.
- `api.py` — FastAPI application with two endpoints:
  - `POST /upload` accepts a single `.md` or `.txt` file and creates a
    temporary per-upload Chroma store under `chroma_tmp/{session_id}`.
    It returns a `session_id` used to query the document.
  - `POST /ask` accepts JSON `{ session_id, question, top_k }` and returns
    the model's answer and sources.
- `compare_embeddings.py` — Small utility to check embeddings and run a
  pairwise embedding distance evaluator.

Prerequisites
-------------
- Python 3.10+ recommended
- An OpenAI API key set in the environment variable `OPENAI_API_KEY` (or a
  `.env` file in the project root). The code uses `python-dotenv` to load a
  `.env` file during development.
- Install required Python packages. Note: `chromadb` may require `onnxruntime`
  which can be tricky to install on some platforms — see notes below.

Install dependencies (basic)
----------------------------
For typical usage install the project's dependencies:

```powershell
pip install -r requirements.txt
pip install python-multipart fastapi uvicorn
pip install "unstructured[md]"
```

If you run into `onnxruntime` installation issues, follow the platform
guidance (conda for macOS, MSVC Build Tools for Windows) before installing
`chromadb`.

Offline: create a persistent vector DB
-------------------------------------
Run the offline indexing pipeline to build a persistent `chroma/` store
from documents in `data/books/`:

```powershell
python create_database.py
```

This will:
- Read `data/books/*.md` using `DirectoryLoader`.
- Split documents into chunks (chunk_size=300, chunk_overlap=100).
- Generate embeddings for each chunk with `OpenAIEmbeddings`.
- Write a Chroma DB to the `chroma/` directory.

Querying the persistent DB (CLI)
--------------------------------
Once the DB exists you can ask questions from the command line:

```powershell
python query_data.py "How does Alice meet the Mad Hatter?"
```

This script will load the `chroma/` store, retrieve top-k chunks, format a
prompt containing only the retrieved context, and call `ChatOpenAI` to get
an answer.

Runtime API (upload & ask)
--------------------------
For an interactive flow where a user uploads a document and then asks
questions against it, start the FastAPI app:

```powershell
# development run (auto-reload)
python api.py
# or using uvicorn explicitly
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Example flow:
1. POST `/upload` with a multipart form file (field name `file`): you will
   receive a JSON response with `session_id`.
2. POST `/ask` with JSON `{ "session_id": "<id>", "question": "..." }`.

The API stores per-upload stores in `chroma_tmp/{session_id}` and writes a
`.meta` file with the upload timestamp so you can implement TTL-based cleanup.

Security and production notes
-----------------------------
- The project is set up for local development and demonstration purposes. For
  production you should add authentication, file-size limits, rate limiting,
  and stricter CORS/allowed origins.
- Per-upload Chroma directories are simple and easy to reason about, but for
  higher scale consider a shared vector DB with namespaced collections.
- Embeddings cost money — be mindful about accepting many uploads or very
  large documents in a public-facing app.

What I changed (comments)
-------------------------
I added docstrings and inline comments to the main Python files so the
purpose and flow of each function is easier to follow. If you'd like, I can
also add brief unit tests or a minimal React frontend to exercise the API.

Troubleshooting
---------------
- If you see import errors for `langchain_community` or `langchain_openai`,
  ensure you have the correct `langchain` and related packages from
  `requirements.txt` installed.
- If the OpenAI client complains about the API key, verify `OPENAI_API_KEY`
  is present in your environment or in a `.env` file at the project root.

- If you see an error like "OpenAIEmbeddings __init__() got an unexpected keyword argument 'proxies'":
  - This usually means the installed versions of `langchain` / `langchain-openai` in your environment
    are incompatible with one another or with the code in this repo. The `proxies` kwarg is not accepted
    by the pydantic model used by `OpenAIEmbeddings` in some package versions.
  - Fixes:
    1. Install the pinned requirements from `requirements.txt` in a fresh virtualenv: `pip install -r requirements.txt`.
    2. Do not pass `proxies` directly to `OpenAIEmbeddings()`; instead configure HTTP(S) proxies via the
       standard environment variables `HTTP_PROXY` / `HTTPS_PROXY` or configure the `openai` client directly:

```powershell
# Windows PowerShell example - set for the current session
$env:HTTPS_PROXY = 'http://proxy.example:3128'
# or set permanently in system environment variables
```

    3. Alternatively, configure the `openai` client if you need a programmatic proxy (check `openai` docs for
       your client version). After configuring the client, call `OpenAIEmbeddings()` without passing `proxies`.

  - If the error persists, check the server logs (`app.log`) for package versions (the app logs langchain and
    langchain-openai versions when embedding creation fails) and share them so we can suggest an exact fix.

Resources
---------
- RAG + LangChain tutorial video used by this demo: 
  https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami

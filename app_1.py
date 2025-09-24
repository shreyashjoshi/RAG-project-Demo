from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import langchain_community
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import shutil

load_dotenv()
# OPENAI_API_KEY no longer required; using local HuggingFace embeddings only.
app = Flask(__name__, template_folder='frontend')

CHROMA_PATH = "chroma"

DATA_PATH = "data/books"

def generate_data_store():
    """Top-level orchestration for building the vector store."""
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    """Load text and markdown documents from DATA_PATH.

    Returns:
      list[Document]: a list of langchain Document objects containing the
      file content and basic metadata (such as the source path).
    """
    documents: list[Document] = []
    # Load markdown files
    md_loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents.extend(md_loader.load())
    # Load plain text files
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents.extend(txt_loader.load())
    return documents


def split_text(documents: list[Document]):
    """Split documents into smaller overlapping chunks suitable for embedding.

    Why split?
      - Large documents produce embeddings that may dilute local context.
      - Smaller chunks allow retrieval to return highly relevant snippets.

    Parameters:
      documents: list of langchain Document objects

    Returns:
      list[Document]: each Document is a chunk with metadata preserved.
    """
    # These parameters strike a balance between chunk length and overlap.
    # Adjust chunk_size / chunk_overlap depending on the domain and expected queries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print an example chunk to help during development. Remove or reduce
    # this in production to avoid leaking PII in logs.
    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """Embed chunks and persist them to a Chroma vector store on disk.

    Steps:
      - Remove any existing CHROMA_PATH directory to start from a clean state.
      - Create a new Chroma DB using OpenAIEmbeddings and persist it.

    Caution:
      - Embedding many documents will call the OpenAI API and may incur cost.
    """
    # Clear out the database first to avoid mixing old and new embeddings.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents and persist it.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def load_chroma_db():
    """Load the existing Chroma database from disk."""
    if not os.path.exists(CHROMA_PATH):
        return None
    
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print(file)
        if file:
            filename = file.filename
            print(filename)
            file.save(os.path.join(DATA_PATH, filename))
            generate_data_store()
            os.remove(os.path.join(DATA_PATH, filename))
            return jsonify({
                'message': 'File uploaded successfully and embeeding done',
                'dataset_id': 'default'  # Since you're using a single global vector store
            }), 200
        else:
            return jsonify({'error': 'No file provided'}), 400
    
    return jsonify({'message': 'Upload endpoint - use POST to upload files'}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer questions about uploaded documents using RAG."""
    try:
        # Get JSON data from request
        data = request.get_json()
        print(f"Received data: {data}")  # Debug: Print received data
        
        if not data or 'question' not in data:
            print("Error: No question found in request data")  # Debug
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question']
        print(f"Question received: {question}")  # Debug: Print the question
        
        # Load the Chroma database
        db = load_chroma_db()
        if not db:
            return jsonify({'error': 'No documents found. Please upload a document first.'}), 404
        
        # Search the DB using the same method as query.py
        results = db.similarity_search_with_relevance_scores(question, k=3)
        
        if len(results) == 0 or results[0][1] < 0.2:
            return jsonify({'answer': 'Unable to find matching results. Please try a different question or upload more relevant documents.'}), 200
        
        # Build context from retrieved documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Create prompt template (same as query.py)
        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}
        """
        
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question)
        
        # For now, return the context as the answer (like query.py does)
        # You can replace this with an actual LLM call later
        answer = context_text[:500] + "..." if len(context_text) > 500 else context_text
        
        # Get sources
        sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'relevance_scores': [score for _doc, score in results]
        })
        
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        return jsonify({'error': f'Failed to process question: {str(e)}'}), 500


@app.route('/')
def index():
    return render_template('index_1.html')




if __name__ == '__main__':
    # In a container we must listen on all interfaces. Disable debug for production.
    app.run(host="0.0.0.0", port=5000, debug=False)
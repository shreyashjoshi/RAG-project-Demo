"""Build a Chroma vector store from local markdown documents.

This script is the offline pipeline used to create a persistent vector
database from documents stored on disk. It's intentionally small and
easy to follow so you can adapt chunk sizes, add more loaders, or swap
the vector store implementation.

High level steps:
  1. Load documents from DATA_PATH using a DirectoryLoader
  2. Split documents into overlapping chunks using a RecursiveCharacterTextSplitter
  3. Embed the chunks using OpenAIEmbeddings and persist them to disk with Chroma

Notes:
  - The OpenAI API key is read from the environment variable OPENAI_API_KEY.
  - Re-running the script removes the existing `chroma` directory so you
    always get a clean rebuild (this is convenient during development).
"""

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil


# Load environment variables from .env (if present). This keeps secrets out
# of source control and enables local config.
load_dotenv()

# Configure OpenAI client. The langchain OpenAI wrappers also read the same
# environment settings, but setting openai.api_key directly ensures any direct
# calls use the configured key.
openai.api_key = os.environ.get("OPENAI_API_KEY")


# Where the persistent Chroma DB will be stored. The query script expects
# the vector store to exist at this path so we keep the default here.
CHROMA_PATH = "chroma"

# Path where source markdown files are stored. DirectoryLoader will recursively
# find files that match the glob pattern.
DATA_PATH = "data/books"


def main():
    """Main entrypoint when executed directly.

    This simply orchestrates the pipeline by calling smaller helper functions.
    """
    generate_data_store()


def generate_data_store():
    """Top-level orchestration for building the vector store."""
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    """Load markdown documents from DATA_PATH.

    Returns:
      list[Document]: a list of langchain Document objects containing the
      file content and basic metadata (such as the source path).
    """
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
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
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")



if __name__ == "__main__":
    main()

import os
import glob
from typing import List
from dotenv import load_dotenv
import uuid
import qdrant_client as qc
import qdrant_client.http.models as qmodels
from langchain.embeddings import LlamaCppEmbeddings

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

load_dotenv()

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

load_dotenv()

client = qc.QdrantClient(url="localhost", port=6333
                                         #path='C:/Users/user/KnowledgeBot/db'
                                         )
METRIC = qmodels.Distance.DOT
DIMENSION = 1024
COLLECTION_NAME = "myDocument"
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
n_gpu_layers = os.environ.get('MODEL_N_GPU')
n_threads = os.environ.get('MODEL_THREAD')

#embeddings = LlamaCppEmbeddings(model_path=model_path, n_ctx=model_n_ctx, n_gpu_layers=n_gpu_layers ,n_threads=n_threads)
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

def create_index():
    client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config = qmodels.VectorParams(
            size=DIMENSION,
            distance=METRIC,
        )
    )

def embed_text(text):
    #embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vectors = embeddings.client.encode(text).tolist()
    #vectors = embeddings.embed_documents([text])
    return vectors

def create_subsection_vector(section_content):
    vector = embed_text(section_content.page_content)
    id = str(uuid.uuid1().int)[:32]
    payload = section_content
    return id, vector, payload

def add_doc_to_index(subsections):
    ids = []
    vectors = []
    payloads = []
    
    for section_content in subsections:
        id, vector, payload = create_subsection_vector(
                section_content
            )
        ids.append(id)
        vectors.append(vector)
        payloads.append(payload)
    
    ## Add vectors to collection
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=qmodels.Batch(
            ids = ids,
            vectors = vectors,
            payloads = payloads
        ),
    )

def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]


def main():
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    chunk_size = 800
    chunk_overlap = 200
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create embeddings

    create_index()

    add_doc_to_index(texts)
        


if __name__ == "__main__":
    main()

import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCUMENTOS_DIR = os.path.join(
    BASE_DIR,
    'chatbot_project',
    'media',
    'documentos_bot'
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    'chatbot_project',
    'chroma_db'
)


def cargar_documentos():

    documentos = []

    for archivo in os.listdir(DOCUMENTOS_DIR):

        ruta = os.path.join(DOCUMENTOS_DIR, archivo)

        if archivo.endswith(".txt"):
            loader = TextLoader(ruta, encoding="utf-8")
            documentos.extend(loader.load())

        elif archivo.endswith(".pdf"):
            loader = PyPDFLoader(ruta)
            documentos.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documentos)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("Documentos cargados correctamente.")


if __name__ == "__main__":
    cargar_documentos()
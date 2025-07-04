from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json

class VectorStore:
    def __init__(self):
        self.faiss_index = None

    def create_store(self, embedder):
        folder1 = "/Users/shivalasya/Desktop/chatbot/data/statements"
        folder2 = "/Users/shivalasya/Desktop/chatbot/data/editorial"
        folder3 = "/Users/shivalasya/Desktop/chatbot/data/metadata_n_"

        files1 = sorted([file for file in os.listdir(folder1) if file.endswith(".txt")])
        files2 = sorted([file for file in os.listdir(folder2) if file.endswith(".txt")])
        files3 = sorted([file for file in os.listdir(folder3) if file.endswith(".json")])

        documents = []

        for file1, file2, file3 in zip(files1, files2, files3):
            page_content = ""
            with open(os.path.join(folder1, file1), "r", encoding="utf-8") as f1, \
                 open(os.path.join(folder2, file2), "r", encoding="utf-8") as f2, \
                 open(os.path.join(folder3, file3), "r", encoding="utf-8") as f3:

                page_content += f1.read() + "\n"
                page_content += f2.read() + "\n"

                # Load metadata and handle both list and dict cases
                loaded_metadata = json.load(f3)

                if isinstance(loaded_metadata, list):
                    if all(isinstance(item, dict) for item in loaded_metadata) and len(loaded_metadata) > 0:
                        metadata = loaded_metadata[0]  # Safely use first dictionary
                    else:
                        raise ValueError(f"Metadata file {file3} is a list but not a valid list of dicts.")
                elif isinstance(loaded_metadata, dict):
                    metadata = loaded_metadata
                else:
                    raise ValueError(f"Metadata file {file3} is not a valid dictionary or list of dicts.")

                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100
        )

        splits = text_splitter.split_documents(documents)
        self.faiss_index = FAISS.from_documents(splits, embedder)
        return self.faiss_index

    def save_store(self, path):
        self.faiss_index.save_local(path)

    def load_store(self, path, embedder):
        self.faiss_index = FAISS.load_local(path, embedder)
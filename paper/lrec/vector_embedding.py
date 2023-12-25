import os
from os import PathLike
from pathlib import Path
from typing import Union, List, Dict, Optional, Literal
import glob
import json
import time
from tqdm import tqdm
from collections import deque

from langchain.document_loaders import TextLoader
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)  # 무료, OpenAIEmbeddings  #유료
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS, Pinecone, Chroma  # 무료
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import RetrievalQA

# __import__("pysqlite3")
# import sys
#
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import pinecone
import chromadb

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class HeritageSplitter:
    def __init__(self, chunk_size: int = 1100, chunk_overlap: int = 0) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_from_file(self, txt_file: Union[str, PathLike, Path]) -> List[Document]:
        json_file = txt_file.replace(".txt", ".json")
        with open(txt_file, "r", encoding="utf-8") as f:
            full_text = "".join(f.readlines())
            # full_text = "\n".join(f.readlines())
        with open(json_file, "r", encoding="utf-8") as f:
            json_obj = json.load(f)
            # json_obj = json_obj["기본정보"]
        txt_file = Path(txt_file)
        return self.split_text(full_text, txt_file.stem)

    def split_text(
            self, text: str, heritage_name: str, metadata: Optional[Dict] = None
    ) -> List[Document]:
        joes = text.split("\n\n")
        docs: List[Document] = list()
        cache = heritage_name
        for jo in joes:
            if len(heritage_name + jo) > self.chunk_size:
                if len(cache) > 0:
                    docs.append(Document(page_content=cache, metadata=metadata))
                    cache = heritage_name
                docs.append(Document(page_content=heritage_name + jo, metadata=metadata))
            else:
                cache += jo
        if len(cache) > len(heritage_name):
            docs.append(Document(page_content=cache, metadata=metadata))
        return docs

def embed_with_chroma(
        # collectoin_name: Literal["law","precedent"],
        persist_directory: Union[str, PathLike, bytes] = "./chroma",
        law_path="heritage_data/",
        # prec_path="law_data/cases",
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
):
    embedding_model = OpenAIEmbeddings()

    heritage_db = Chroma(
        collection_name="heritage",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )
    heritage_splitter = HeritageSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    lq = deque()
    heritage_txt = "./서울시 유적지 현황 (영어).json"
    heritage_docs = heritage_splitter.split_from_file(heritage_txt)
    
    try:
        heritage_db.add_documents(documents=heritage_docs)
    except:
        lq.append(heritage_docs)

def embed_with_pinecone(
        root_path: Union[str, PathLike, bytes] = "./heritage_data",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
):
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
    )

    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                txt_files.append(file_path)

    txt_spliter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    embedding_model = OpenAIEmbeddings()
    vector_store = Pinecone(
        index=pinecone.Index(os.getenv("PINECONE_INDEX")),
        embedding=embedding_model,
        text_key="text",
    )
    for txt_file in txt_files:
        raw_docs = TextLoader(txt_file).load()
        splited_docs = txt_spliter.split_documents(raw_docs)
        vector_store.add_documents(splited_docs)


def get_pinecone_retriever() -> VectorStoreRetriever:
    db_call = Pinecone(
        index=pinecone.Index(os.getenv("PINECONE_INDEX")),
        embedding=OpenAIEmbeddings(),
        text_key="text",
    )
    return db_call.as_retriever()


def get_chroma_retreiver(
        collection_name: Literal["heritage"], persist_directory: str = "./chroma"
):
    db = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
    return db.as_retriever()


if __name__ == "__main__":
    embed_with_chroma()
    heritage_retriever = get_chroma_retreiver(
        collection_name="heritage",
    )
    while True:
        q = input("query:")
        heritage_docs = heritage_retriever.get_relevant_documents(q)
        print(heritage_docs)

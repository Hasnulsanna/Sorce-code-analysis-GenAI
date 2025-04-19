from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
from pathlib import Path


#1. Clone github repo

def repo_ingestion(repo_url):
    print(os)
    print(os.mkdir)
    os.mkdir("reposi", exist_ok=True)
    repo_path = "reposi/"
    Repo.clone_from(repo_url,to_path=repo_path)



#2. Load repo as documents

def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON,parser_threshold=500)
    )
    documents = loader.load()
    return documents

#3. Create chunks

def text_splitter(documents):
    document_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = document_splitter.split_documents(documents)
    return text_chunks

#4. Load Embedding model

def load_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

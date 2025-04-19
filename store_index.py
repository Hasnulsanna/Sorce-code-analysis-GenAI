from src.helper import *
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()


documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()


vectordb = Chroma.from_documents(
    text_chunks,embedding=embeddings,persist_directory="./db"
)
vectordb.persist()
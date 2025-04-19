import os
from dotenv import load_dotenv
from src.helper import *
from flask import Flask,render_template,jsonify,request
from langchain.memory import ConversationSummaryMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate



app = Flask(__name__)

#load env
load_dotenv()

#load embedding model
embeddings = load_embedding()

#load data from vectorstore: db 
persist_directory = "db"
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)


#create llm
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3, max_tokens=500)

#create memory
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)

#create chain
from langchain.chains import ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever = vectordb.as_retriever(search_type="mmr",search_kwargs={"k":8}),
    memory=memory
)

#create routes
@app.route("/",methods = ['GET','POST'])
def index():
    return render_template("index.html")


@app.route("/chatbot",methods =["GET","POST"])
def gitRepo():
    if request.method =='POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")
    return jsonify({"response":str(user_input)})

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == 'clear':
        os.system("rm -rf repo")
    
    result = qa(input)
    print(result["answer"])
    return str(result["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)




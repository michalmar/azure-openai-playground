import streamlit as st
from urllib.error import URLError
import pandas as pd
from utilities import utils, translator
import os



import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from langchain.document_loaders import WebBaseLoader

from dotenv import load_dotenv
load_dotenv()


def initialize(engine='davinci'):
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Initialize gpt-35-turbo and our embedding model
    # TODO -> model name should be a parameter
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", openai_api_version="2023-03-15-preview")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)


    return (llm, embeddings)


def load_page(embeddings, llm):
    # loader = DirectoryLoader('.', glob="Strecha.txt", loader_cls=TextLoader)
    loader = WebBaseLoader(st.session_state['web_url'])
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(documents=docs, embedding=embeddings)

    # Adapt if needed
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=db.as_retriever(),
                                            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=False)

    return qa

def get_response(qa, question):
    query = question
    # query = "Jak ušetřit energii?"
    result = qa({"question": query, "chat_history": st.session_state['messages']})

    return result

def set_url():
    # st.write(f'URL: {st.session_state['web_url']}')
    st.session_state["web_url"] =  st.session_state["url_input"]
    pass

# df = utils.initialize(engine='davinci')
(llm, embeddings) = initialize()

def display_conversation():
    if (len(st.session_state['messages']) > 0):
        with st.expander("History"):
            for message in st.session_state['messages']:
                st.caption(f":blue[Question:]: {message[0]}")
                st.caption(f":blue[Answer:]: {message[1]}")

try:

    default_prompt = "" 
    default_question = "" 
    default_answer = ""
    default_web = ""
    messages = []

    if 'question' not in st.session_state:
        st.session_state['question'] = default_question
    if 'web_url' not in st.session_state:
        st.session_state['web_url'] = ""
    if 'qa' not in st.session_state:
        st.session_state['qa'] = ""
    if 'kbinit' not in st.session_state:
        st.session_state['kbinit'] = False
    if 'messages' not in st.session_state:
        st.session_state['messages'] = messages
        # add_message("system", st.session_state['system'])
    
    #  Set page layout to wide screen and menu item
    menu_items = {
	'Get help': None,
	'Report a bug': None,
	'About': '''
	 ## Embeddings App
	 Embedding testing application.
	'''
    }
    st.set_page_config(layout="wide", menu_items=menu_items, initial_sidebar_state="collapsed")

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.image(os.path.join('images','microsoft.png'))
    with col3:
        with st.expander("Settings"):
            model = st.selectbox(
                "OpenAI Model",
                (os.environ['OPENAI_CHAT_ENGINES'].split(','))
            )
            st.text_area("System",height=100, key='system')
            st.tokens_response = st.slider("Tokens response length", 100, 2000, 1000)
            st.temperature = st.slider("Temperature", 0.0, 1.0, 0.8)
    
    if (not st.session_state['kbinit']):
        
        # occures when nothing is selected -> no web page
        if (st.session_state['web_url'] == ""):
            st.text_input("Enter URL (format: https://www.espn.com/)", key="url_input", on_change=set_url)

        # works only if web is selected but not crawled yet
        if (st.session_state['web_url'] != ""):
            st.session_state["qa"] = load_page(embeddings, llm)
            st.session_state['kbinit'] = True

    # works only if web has been crawled
    if (st.session_state['kbinit']):
        st.caption(f"Page {st.session_state['web_url']} loaded - you may ask questions.")
        question = st.text_input("Ask question based on selected web page content", default_question)
        if (question != ""):
            result = get_response(st.session_state["qa"], question)

            # st.write("Question:", question)
            st.write(":blue[Answer]:", result["answer"])

            st.session_state['messages'].append((question, result["answer"]))


    st.caption(f"To clear the conversation, refresh the page | current model: **{model}**")

    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        display_conversation()


except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
        """
        % e.reason
    )
import streamlit as st
from urllib.error import URLError
import pandas as pd
from utilities import utils, translator
import os


import openai

from dotenv import load_dotenv
load_dotenv()


def initialize(engine='davinci'):
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_key = os.getenv("OPENAI_API_KEY")

# df = utils.initialize(engine='davinci')
df = initialize()

# Defining a function to send the prompt to the ChatGPT model
def send_message(messages, model_name=  "gpt-35-turbo", max_response_tokens=500):
    
    # model_name = "gpt-35-turbo"
    # model_name = "gpt-4"


    response = openai.ChatCompletion.create(
    engine=model_name, 
    messages=messages,
    temperature=0.5,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)

    # chatGPT api
    # tmp_response  = response['choices'][0]['text'].strip()

    # GPT-4 api
    tmp_response = response['choices'][0]["message"]["content"].strip()

    # {"model": model_name, "response": tmp_response}
    return tmp_response

base_system_message = """You are a marketing writing assistant. You help come up with creative content ideas and content like marketing emails, blog posts, tweets, ad copy, listicles, product FAQs, and product descriptions. 
You write in a friendly yet professional tone and you can tailor your writing style that best works for a user-specified audience. 

Additional instructions:
- Make sure you understand your user's audience so you can best write the content.
- Ask clarifying questions when you need additional information. Examples include asking about the audience or medium for the content.
- Don't write any content that could be harmful.
- Don't write any content that could be offensive or inappropriate.
- Don't write any content that speaks poorly of any product or company.
"""


def display_conversation():
    # print(f"Conversations - model: {model} ")
    with st.container():
        for message in st.session_state['messages']:

            if (message['role'] == "assistant"):
                st.write(f":red[{message['role']}({model})]: {message['content']}")
            elif (message['role'] == "user"):
                st.write(f":blue[{message['role']}]: {message['content']}")
            elif (message['role'] == "system"):
                # do not print system messages
                pass
            # st.write("---")

# potential roles: system, user, assistant
def add_message(who, msg):
    if (who == "system"):
        st.session_state['messages'].append({"role": "system", "content": msg})
    elif (who == "user"):
        st.session_state['messages'].append({"role": "user", "content": msg})
    elif (who == "assistant"):
        st.session_state['messages'].append({"role": "assistant", "content": msg})
    else:
        print(f"ERROR: add_message() - who is not valid: {who}")

def ask_bot(model):
    add_message("user", st.session_state['question'])
    
    response = send_message(messages = st.session_state['messages'], model_name=model)
    add_message("assistant", response)
    
    return True

# not used
def clear_conversation():
    st.session_state['messages'] = []
    st.session_state.clear()
    st.session_state['question'] = default_question
    print("conversation cleared")



# df = utils.initialize(engine='davinci')

# @st.cache_data(suppress_st_warning=True)
def get_languages():
    return translator.get_available_languages()

try:

    default_prompt = "" 
    default_question = "" 
    default_answer = ""
    messages = []

    if 'question' not in st.session_state:
        st.session_state['question'] = default_question
    if 'system' not in st.session_state:
        st.session_state['system'] = base_system_message.replace(r'\n', '\n')
    if 'limit_response' not in st.session_state:
        st.session_state['limit_response'] = True
    if 'full_prompt' not in st.session_state:
        st.session_state['full_prompt'] = ""
    if 'messages' not in st.session_state:
        st.session_state['messages'] = messages
        add_message("system", st.session_state['system'])
    
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

    # Get available languages for translation
    available_languages = get_languages()

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.image(os.path.join('images','microsoft.png'))

    col1, col2, col3 = st.columns([2,2,2])
    with col3:
        with st.expander("Settings"):
            model = st.selectbox(
                "OpenAI Model",
                (os.environ['OPENAI_CHAT_ENGINES'].split(','))
            )
            st.text_area("System",height=100, key='system')
            st.tokens_response = st.slider("Tokens response length", 100, 2000, 1000)
            st.temperature = st.slider("Temperature", 0.0, 1.0, 0.8)
    
    # question = st.text_input("Chat", default_question)
    question = st.text_area(label="Chat", height=100)

    st.caption(f"To clear the conversation, refresh the page | current model: **{model}**")
    # st.button("Ask", on_click=ask_bot)
    # st.button("Clear conversation", on_click=clear_conversation)
    
    if question != '':
        st.session_state['question'] = question
        ask_bot(model)
        display_conversation()

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
        """
        % e.reason
    )
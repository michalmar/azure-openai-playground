import streamlit as st
from urllib.error import URLError
import pandas as pd
from utilities import utils
import os
from utilities.azureblobstorage import  get_all_prompt_files

df = utils.initialize(engine='davinci')

def clear_summary():
    st.session_state['summary'] = ""

def get_custom_prompt():
    prompt_text = st.session_state['prompt_text']
    customprompt = "{}".format(prompt_text)
    return customprompt

def customcompletion():
    _, response = utils.get_completion(get_custom_prompt(), max_tokens=st.tokens_response, model=os.getenv('OPENAI_ENGINES', 'text-davinci-003'), temperature=st.temperature)
    
    # st.session_state['memory'].append({"question": st.session_state['prompt_text'],"prompt": get_custom_prompt(), "response": response['choices'][0]['text'].encode().decode()})
    st.session_state['memory'].append({"prompt": get_custom_prompt(), "response": response['choices'][0]['text'].encode().decode()})
    
    st.session_state['result'] = response['choices'][0]['text'].encode().decode()
    st.session_state['response'] = response['usage']

def display_memory():
    
    with st.expander("History"):
        if (len(st.session_state['memory']) > 0):
            # st.text_area(label="Memory", value=st.session_state['memory'], height=200)
            st.json(st.session_state['memory'])
        else:
            st.caption("Empty")

try:
    # Set page layout to wide screen and menu item
    menu_items = {
    'Get help': None,
    'Report a bug': None,
    'About': '''
     ## Embeddings App
     Embedding testing application.
    '''
    }
    st.set_page_config(layout="wide", menu_items=menu_items)

    # check whether the prompts variable exists, if not fetch prompts from the storage
    if 'prompts' not in st.session_state:
        # print("prompts does not exist, fetching from storage")
        st.session_state['prompts'] = {
                                        "None (make your own from scratch)": "",
                                    }
        # load the prompts - prompts are stored in a txt files, on azure blob storage
        prompts_data = get_all_prompt_files()

        for p in prompts_data:
            st.session_state['prompts'][p["prompt_name"]] = p["prompt"]
    else:
        # print("prompts exists, not fetching from storage")
        pass
     
    prompts = st.session_state['prompts']

    if 'memory' not in st.session_state:
        st.session_state['memory'] = []

    if 'example' not in st.session_state:
        st.session_state['example'] = ""

    st.markdown("## Bring your own prompt")
    st.caption("known issue: when changing the prompt, the generated response is not updated or text ['Not in the text.'] appears. Please generate again to fix.")

    st.tokens_response = st.slider("Tokens response length", 100, 1000, 400)
    st.temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    # st.selectbox("Language", [None] + list(available_languages.keys()), key='translation_language')
    
    # parse dictionary of prompts yielding a list keys only
    example = st.selectbox(
                label="Examples",
                options=list(list(prompts.keys()))
            )

    # displaying a box for a custom prompt
    st.session_state['prompt_text'] = st.text_area(label="Prompt", key='prompt', height=400, value=prompts[example])
    st.button(label="Generate", on_click=customcompletion)
    
    # displaying the summary
    st.markdown("**Generated response**")
    
    result = ""
    if 'result' in st.session_state:
        result = st.session_state['result']
    # st.text_area(label="OpenAI result", value=result, height=200)
    
    response = "None"
    if 'response' in st.session_state:
        response = st.session_state["response"]
        st.write(result)
        st.write("Response details:")
        st.write(response)
    
    display_memory()

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
        """
        % e.reason
    )
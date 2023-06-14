from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from urllib.error import URLError
import pandas as pd
from utilities import utils, translator
import os

@st.cache(suppress_st_warning=True)

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
    st.set_page_config(layout="wide", menu_items=menu_items, initial_sidebar_state="collapsed")

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.image(os.path.join('images','microsoft.png'))

    st.title(f"Hello user <placehoder>!")






except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
        """
        % e.reason
    )
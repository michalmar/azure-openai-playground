import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os, io, zipfile
from tenacity import retry, wait_random_exponential, stop_after_attempt
from transformers import GPT2Tokenizer
from utilities.redisembeddings import execute_query, get_documents, set_document
from utilities.formrecognizer import analyze_read
from utilities.azureblobstorage import upload_file, upsert_blob_metadata
import tiktoken
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def initialize(engine='davinci'):
    openai.api_type = "azure"
    # openai.api_version = "2023-03-15-preview"
    openai.api_version = "2022-12-01"
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_key = os.getenv("OPENAI_API_KEY")


# Semantically search using the computed embeddings locally
def search_semantic(df, search_query, n=3, pprint=True, engine='davinci'):
    embedding = get_embedding(search_query, engine= get_embeddings_model()['query'])
    df['similarities'] = df[f'{engine}_search'].apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n)
    if pprint:
        for r in res:
            print(r[:200])
            print()
    return res.reset_index()

# Semantically search using the computed embeddings on RediSearch
def search_semantic_redis(df, search_query, n=3, pprint=True, engine='davinci'):
    embedding = get_embedding(search_query, engine= get_embeddings_model()['query'])
    res = execute_query(np.array(embedding))

    if pprint:
        for r in res:
            print(r[:200])
            print()
    return res.reset_index()

# Return a semantically aware response using the Completion endpoint
def get_semantic_answer(df, question, explicit_prompt="", model="DaVinci-text", engine='babbage', limit_response=True, tokens_response=100, temperature=0.0):

    restart_sequence = "\n\n"
    question += "\n"

    res = search_semantic_redis(df, question, n=3, pprint=False, engine=engine)

    if len(res) == 0:
        prompt = f"{question}"
    elif limit_response:
        res_text = "\n".join(res['text'][0:int(os.getenv("NUMBER_OF_EMBEDDINGS_FOR_QNA",1))])
        question_prompt = explicit_prompt.replace(r'\n', '\n')
        question_prompt = question_prompt.replace("_QUESTION_", question)
        prompt = f"{res_text}{restart_sequence}{question_prompt}"
    else:
        prompt = f"{res_text}{restart_sequence}{question}"
            

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=tokens_response,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    print(f"{response['choices'][0]['text'].encode().decode()}\n\n\n")

    return prompt,response#, res['page'][0]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, engine="text-embedding-ada-002") -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    EMBEDDING_ENCODING = 'cl100k_base' if engine == 'text-embedding-ada-002' else 'gpt2'
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    return openai.Embedding.create(input=encoding.encode(text), engine=engine)["data"][0]["embedding"]

def split_and_embed(text: str, filename="", chunk_size=1500, separator=" ", engine="text-embedding-ada-002"):
    # Here we split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, separator=separator)
    docs = []
    metadatas = []
    full_data_arr = []
    
    splits = text_splitter.split_text(text)

    docs.extend(splits)
    metadatas.extend([{"source": filename}] * len(splits))

    # for each document in docs, convert to bytes and upload to Azure Blob Storage
    for i, doc in enumerate(docs):
        doc_bytes = doc.encode("utf-8")
        doc_filename = f"{filename}_part{i:03d}.txt"

        upload_file(doc_bytes, doc_filename)
        # upsert_blob_metadata(doc_filename, metadatas[i])
        
        full_data = {
            "text": doc,
            "filename": doc_filename,
            "search_embeddings": get_embedding(doc, engine)
        }
        full_data_arr.append(full_data)

        upsert_blob_metadata(doc_filename, { 'embeddings_added': 'true'})
    return full_data_arr




def chunk_and_embed(text: str, filename="", engine="text-embedding-ada-002"):
    EMBEDDING_ENCODING = 'cl100k_base' if engine == 'text-embedding-ada-002' else 'gpt2'
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)

    full_data = {
        "text": text,
        "filename": filename,
        "search_embeddings": None
    }

    text = text.replace("\n", " ")
    lenght = len(encoding.encode(text))
    if engine == 'text-embedding-ada-002' and lenght > 2000:
        return split_and_embed(text, filename, chunk_size=1500, separator=" ", engine=engine)
    elif lenght > 3000:
        return split_and_embed(text, filename, chunk_size=2500, separator=" ", engine=engine)

    full_data['search_embeddings'] = get_embedding(text, engine)

    return full_data


def get_completion(prompt="", max_tokens=400, model="text-davinci-003", temperature=1):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    print(f"{response['choices'][0]['text'].encode().decode()}\n\n\n")

    return prompt,response#, res['page'][0]


def get_token_count(text: str):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return len(tokenizer(text)['input_ids'])


def get_embeddings_model():
    OPENAI_EMBEDDINGS_ENGINE_DOC = os.getenv('OPENAI_EMEBDDINGS_ENGINE', os.getenv('OPENAI_EMBEDDINGS_ENGINE_DOC', 'text-embedding-ada-002'))  
    OPENAI_EMBEDDINGS_ENGINE_QUERY = os.getenv('OPENAI_EMEBDDINGS_ENGINE', os.getenv('OPENAI_EMBEDDINGS_ENGINE_QUERY', 'text-embedding-ada-002'))
    return {
        "doc": OPENAI_EMBEDDINGS_ENGINE_DOC,
        "query": OPENAI_EMBEDDINGS_ENGINE_QUERY
    }


def add_embeddings(text, filename, engine="text-embedding-ada-002"):
    embeddings = chunk_and_embed(text, filename, engine)
    if embeddings:
        # Store embeddings in Redis
        if isinstance(embeddings, list):
            for e in embeddings:
                set_document(e)
        else:
            set_document(embeddings)
    else:
        st.error("No embeddings were created for this document as it's too long. Please keep it under 3000 tokens")


def convert_file_and_add_embeddings(fullpath, filename, enable_translation=False):
    # Extract the text from the file
    text = analyze_read(fullpath)
    # Upload the text to Azure Blob Storage
    zip_file = io.BytesIO()
    if enable_translation:
        text = list(map(lambda x: translate(x), text))
    for k, v in enumerate(text):
        with zipfile.ZipFile(zip_file, mode="a") as archive:
            archive.writestr(f"{k}.txt", v)
    upload_file(zip_file.getvalue(), f"converted/{filename}.zip", content_type='application/zip')
    upsert_blob_metadata(filename, {"converted": "true"})
    for k, t in enumerate(text):
        add_embeddings(t, f"{filename}_chunk_{k}", os.getenv('OPENAI_EMBEDDINGS_ENGINE_DOC', 'text-embedding-ada-002'))
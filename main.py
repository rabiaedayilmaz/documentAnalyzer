import streamlit as st

import faiss
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
import pickle
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


st.title('Albert Kullanım Kılavuzuna Sor')

openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_org_id = st.secrets["OPENAI_ORGANIZATION_KEY"]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

@st.cache_data
def load_data(path="guidelines_data/docs.index"):
    index = faiss.read_index(path)
    with open("guidelines_data/faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
        return store, index

vectorstore, index = load_data()
vectorstore.index = index

template = """"You are Albert, a voice-based health assistant designed to provide assistance and support to users.
You talk sincerely and help others. You can only speak in Turkish.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Üzgünüm, bu konuda bir bilgim yok." Don't try to make up an answer.
If the question is not about the Albert Health App Guidelines, politely inform them that you are tuned to only answer questions about the Albert Health App Guidelines.
Question: {question}
=========
{context}
=========
Answer:"""
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the Albert Health App Guidelines.
Chat History:
{chat_history}
Follow Up Input: {question}
{template}
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
i=0

def get_chain(vectorstore, stream_handler):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.1, streaming=True, callbacks=[stream_handler])
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
    )
    return qa_chain

def generate_response(input_text):
    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box)
    qa_chain = get_chain(vectorstore, stream_handler)
    result = qa_chain({"question": input_text, "chat_history": st.session_state.chat_history})
    return result

with st.form('my_form'):
    text = st.text_area('Sorularını dinliyorum...', 'Albert, senin ile neler yapabilirim?')
    col1, col2 = st.columns([3, 1])
    with col1:
        submitted = st.form_submit_button('Gönder \U0001F914')
    if submitted and openai_api_key.startswith('sk-'):
        i += 1
        result = generate_response(text)
        last_chat = {
        "input_text": result["question"],
        "answer": result['answer'],
        "source_documents": result['source_documents'],
        }
        st.session_state.chat_history.append(last_chat)
    with col2:
        show_source = st.form_submit_button("\U0001F4C4 Kaynağı Göster")
    if show_source:
        st.markdown(f"Kaynak İçerik:\n\n{st.session_state.chat_history[i]['source_documents'][0].page_content}")

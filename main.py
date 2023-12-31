import streamlit as st

import faiss
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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
    st.session_state.i = 0
    st.session_state.last_chat = {}

@st.cache_data
def load_data(path="guidelines_data/docs.index"):
    index = faiss.read_index(path)
    with open("guidelines_data/faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
        return store, index

vectorstore, index = load_data()
vectorstore.index = index

template = """You are Albert, a voice-based health assistant designed to provide assistance and support to users.
Your responses should be informative and polite. Please communicate in Turkish.
You have access to the Albert Health App Guidelines document for reference.
If you encounter a question that is not related to the Albert Health App Guidelines, politely inform the user that you can only answer questions about the guidelines.
If you don't know the answer to a question, respond with "Üzgünüm, bu konuda bir bilgim yok." (I'm sorry, I don't have information on this).
Now, let's proceed with the user's query:
Question: {question}
=========
{context}
=========
Answer:"""

_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to make it more explicit.
Please assume that the question is about the Albert Health App Guidelines.
Chat History:
{chat_history}
Follow-Up Input: {question}
{template}
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def get_chain(vectorstore, stream_handler):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.1, streaming=True, callbacks=[stream_handler])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False, output_key='answer')
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
    )
    return qa_chain

def generate_response(input_text):
    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box)
    qa_chain = get_chain(vectorstore, stream_handler)
    result = qa_chain({"question": input_text, "chat_history": st.session_state.last_chat})
    st.session_state.i += 1
    return result

with st.form('my_form'):
    text = st.text_area('Sorularını dinliyorum...', 'Albert, senin ile neler yapabilirim?')
    col1, col2 = st.columns([3, 1])
    with col1:
        submitted = st.form_submit_button('Gönder \U0001F914')
    if submitted and openai_api_key.startswith('sk-'):
        result = generate_response(text)
        st.session_state.last_chat = {
        "input_text": result["question"],
        "answer": result['answer'],
        "source_documents": result['source_documents'],
        }
        st.session_state.chat_history.append(st.session_state.last_chat)
    with col2:
        show_source = st.form_submit_button("\U0001F4C4 Kaynağı Göster")
    if show_source:
        st.markdown(f"Kaynak İçerik:\n\n{st.session_state.last_chat['source_documents'][0].page_content}")

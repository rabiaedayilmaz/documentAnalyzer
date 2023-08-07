import streamlit as st

import faiss
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks.base import BaseCallbackHandler
import pickle

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text)


st.title('Albert Kullanım Kılavuzuna Sor')

openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_org_id = st.secrets["OPENAI_ORGANIZATION_KEY"]

@st.cache_data
def load_data(path="guidelines_data/docs.index"):
  index = faiss.read_index(path)
  with open("guidelines_data/faiss_store.pkl", "rb") as f:
      store = pickle.load(f)
      return store, index

store, index = load_data()
store.index = index
def generate_response(input_text):
  question_prompt = f"""
  You are Albert.
  Never forget to answer only in Turkish. 
  Question: {input_text}
  """
  chat_box=st.empty()
  stream_handler = StreamHandler(chat_box)
  chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0.1, streaming=True, callbacks=[stream_handler]), retriever=store.as_retriever())
  result = chain(question_prompt)

with st.form('my_form'):
  text = st.text_area('Sorularını dinliyorum...', 'Albert, seni nasıl kullanabilirim?')
  submitted = st.form_submit_button('Gönder')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)
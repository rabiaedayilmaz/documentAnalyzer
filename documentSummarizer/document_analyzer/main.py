import streamlit as st
from ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from core.parser import read_file
from core.caching import bootstrap_caching
from core.chunking import chunk_file
from core.embedding import embed_files
from core.qa import query_folder
from core.utils import get_llm

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-3.5-turbo", "gpt-4", "debug"]

st.set_page_config(page_title="Albert'a Sor", page_icon="💙", layout="wide")
st.header("Albert'a Sor 💙")

openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_org_id = st.secrets["OPENAI_ORGANIZATION_KEY"]

# enable caching expensive functions
bootstrap_caching()

if not openai_api_key:
    st.warning(
        "OpenAI API anahtarında bir sorun var 👀"
    )

uploaded_file = st.file_uploader(
    "Belge yükleyelim: pdf, docx, txt ve çeşitli görsel formatlarını destekliyorum."
)

model: str = st.selectbox("Model", options=MODEL_LIST)

with st.expander("Gelişmiş Seçenekler"):
    return_all_chunks = st.checkbox("Vektör aramasında bulunan tüm yığınları göster.")
    show_full_doc = st.checkbox("Belgeden ayrıştırılan içerikleri göster.")

if not uploaded_file:
    st.stop()

try:
    file = read_file(uploaded_file)
except Exception as e:
    display_file_read_error(e)

chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

if not is_file_valid(file):
    st.stop()

if not is_open_ai_key_valid(openai_api_key, model):
    st.stop()

with st.spinner("Belge işleniyor... Birazcık bekleteceğim ⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING if model != "debug" else "debug",
        vector_store=VECTOR_STORE if model != "debug" else "debug",
        openai_api_key=openai_api_key,
    )

with st.form(key="qa_form"):
    query = st.text_area("Yüklediğin belge hakkında soru sor!")
    submit = st.form_submit_button("Soruyu Gönder")

if show_full_doc:
    with st.expander("Belge"):
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)

if submit:
    if not is_query_valid(query):
        st.stop()

    answer_col, sources_col = st.columns(2)

    llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        llm=llm,
    )

    with answer_col:
        st.markdown("#### Cevap:")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Kaynak:")
        for source in result.sources:
            print(source)
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
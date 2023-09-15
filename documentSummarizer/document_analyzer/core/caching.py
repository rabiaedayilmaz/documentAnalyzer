import streamlit as st
from streamlit.runtime.caching.hashing import HashFuncsDict

import core.parser as parser
import core.chunking as chunking
import core.embedding as embedding
from core.parser import File

def file_hash_func(file: File) -> str:
    """Get unique hash for a file."""
    return file.id

@st.cache_data(show_spinner=False)
def bootstrap_caching():
    """Patch module functions with caching"""
    file_subtypes = [
        cls for cls in vars(parser).values()
          if isinstance(cls, type) and issubclass(cls, File) and cls != File
          ]
    file_hash_funcs: HashFuncsDict = {cls: file_hash_func for cls in file_subtypes}

    parser.read_file = st.cache_data(show_spinner=False)(parser.read_file)
    chunking.chunk_file = st.cache_data(show_spinner=False, hash_funcs=file_hash_funcs)(chunking.chunk_file)
    embedding.embed_files = st.cache_data(show_spinner=False, hash_funcs=file_hash_funcs)(embedding.embed_files)
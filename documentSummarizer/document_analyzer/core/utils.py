from typing import List
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from core.debug import FakeChatModel


def pop_docs_upto_limit(
        query: str, chain: StuffDocumentsChain, docs: List[Document], max_len: int
) -> List[Document]:
    token_count: int = chain.prompt_length(docs, question=query)
    
    while token_count > max_len and len(docs) > 0:
        docs.pop()
        token_count = chain.prompt_length(docs, question=query)
    return docs

def get_llm(model: str, **kwargs) -> BaseChatModel:
    if model == "debug":
        return FakeChatModel()
    if "gpt" in model:
        return ChatOpenAI(model=model, **kwargs)
    
    raise NotImplementedError(f"Model {model} not supported!")
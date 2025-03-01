import copy
from typing import List, Union
from src.utilize.utilize import *


THINK_TOKEN = "<QUERY>"
SEARCH_TOKEN = "<SEARCH>"
FACT_TOKEN = "<FACT>"
FINAL_TOKEN = "<FINAL>"

PROMPT = f"""You are an intelligent search agent capable of simulating a question-answering process by actively seeking information from Wikipedia to answer a given question.

Specifically, given an open-domain query, please iteratively: (1) Formulate a sub-query to search on Wikipedia; (2) Select useful documents from the search results and (3) Extract supporting facts from the selected documents.
Your output should include three types of special actions corresponding to the above steps:
(1) {THINK_TOKEN}: Formulate a sub-query.
(2) {SEARCH_TOKEN}: Retrieve and carefully read the documents using the formulated sub-query.
(3) {FACT_TOKEN}: Extract the answer to the sub-query from the documents.

Since this is a multi-hop question, your output should interleave {THINK_TOKEN}, {SEARCH_TOKEN}, and {FACT_TOKEN} actions until reaching the final answer. Conclude your output with the special token {FINAL_TOKEN} followed by the final answer.

Below is the task for you to complete:""" + """

<USER QUERY> {task}
Your Output:"""


user_rank_prompt = """You are an intelligent search agent capable of simulating the question-answering process by actively seeking information from Wikipedia to answer the provided question.

Given the formulated query: "{question}", there are {n_docs} relevant documents, each identified by a unique numerical identifier []. The documents are listed below: 
{docs}

Your task is to select the useful documents from the search results based on their usefulness in answering the search query. A passage's usefulness is defined by the following criteria:
1. Relevance to the query.
2. Contains essential information required to answer the query.

You should output only the numeric document identifiers of the selected documents. If more than one document is selected, arrange their identifiers in descending order using the format [] > [], e.g., [2] > [1]. Provide only the selection results and avoid explanations.

Search Query: {question}
Your Output:"""

FINAL_SENTENCE = f'{FINAL_TOKEN}\n{SEARCH_TOKEN} end search. please give the final answer to the input question: ' + '{question}'


def rank_instruction(question: str, candidates: List[str]):
    _docs = copy.deepcopy(candidates)
    _docs = [f'[{i}] {doc}' for i, doc in enumerate(_docs,1)]
    n_docs = len(_docs)
    _docs = '\n'.join(_docs)
    return user_rank_prompt.format(docs=_docs, question=question, n_docs=n_docs)


def get_doc_by_ids(doc_ids: Union[str, List], candidates: List[str], max_id: int = 100):
    if type(doc_ids) == str:
        doc_ids = extract_numbers_from_ordered_brackets(doc_ids)
    assert type(doc_ids) == list
    docs = [candidates[docid] for docid in doc_ids if 0 <= int(docid) < max_id]
    return docs


class TextNormalization(object):
    special_annotate_prefix = ['<Search', '<Reference', '<Keyword']
    def normalize(self, text):
        if text is None:
            return text
        _text = copy.deepcopy(text)
        for i in range(0,10):
            _text = _text.replace(f'{i}>', '')
        for p in TextNormalization.special_annotate_prefix:
            _text = _text.replace(p, '')
        for p in [THINK_TOKEN, SEARCH_TOKEN, FACT_TOKEN, FINAL_TOKEN]:
            _text = _text.replace(p, '')
        return _text.strip()

    def is_search(self, sent):
        return '<Search' in sent

    def is_reference(self, sent):
        return '<Reference' in sent

    def is_keyword(self, sent):
        return '<Keyword' in sent

    def is_final(self, sent):
        return '<Final' in sent

    def formulate(self, idx, query, doc, answer):
        text = f"""<QUERY> {query}
<SEARCH> {doc}
<FACT> {answer}
"""
        return text
# Implement the Retrieval

## Our official retrieval corpus

We follow previous work and use the [Wikipedia 2018](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz) as our document corpus, which can be found in [DPR](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py) repo.

## Our official retrieval model
In our official experiment, we use the [ColBERT](https://github.com/stanford-futuredata/ColBERT/tree/main) as the retrieval model to pair each query or sub-query with top-k (k=20) documents. The pre-trained ColBERT checkpoint can be downloaded in either its official repo or its [link](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz).
Since the ColBERT repo may update, to make the deployment easier, we copy the version of ColBERT that used in our experiment in this repo, e.g., in `src/retrieval/ColBERT` folder.
After download the corpus, you can deploy a out-of-box *ColBERT* model in your experimental environment as following.
1. Documents -> Embeddings: encode the document corpus
```shell
python index.py
```
You can also directly download the encoded embedding used in our experiment. Please click this [link]() to download.
2. Deploy the retrieval through `Flask` library: deploy your retrieval model, making it can be accessed through `python requests`.
```shell
cd src/retrieval/ColBERT

python server_retrieval.py
```
Once the ColBERT is employed, the command terminal will show the `url`. You should set the request url in `os.environment`. For example,
```python
import os
os.environ['API_SEARCH'] = "http://10.96.202.234:8893"
```

## Customize your own retrieval system
If you use your customized retrieval such as Bge retriever, Bing search or Google search, please replace the following retrieval function in `src/utilize/apis.py` with your own retrieval function. In more details, please re-implement the following code snippet in our `src/utilize/apis.py` file:
```python
import requests
import os

api_search = os.getenv("API_SEARCH", "http://10.96.201.177:8893")

def document_retrieval(query, k=20):
    url = f'{api_search}/api/search?query={query}&k={k}'
    response = requests.get(url)
    res = response.json()
    knowledge = []
    for doc in res['topk']:
        text = doc['text'][doc['text'].index('|') + 1:].replace('"','').strip()
        title = doc['text'][:doc['text'].index('|')].replace('"','').strip()
        knowledge.append(f"Title: {title}. Content: {text}")
    return knowledge
```

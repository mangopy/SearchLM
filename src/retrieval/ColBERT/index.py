from typing import overload
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="hotpotqa_wiki")):

        config = ColBERTConfig(
            nbits=2,
            overwrite=True,
            root="/root/paddlejob/workspace/env_run/output/ColBERT/experiments",
        )
        indexer = Indexer(checkpoint="/root/paddlejob/workspace/env_run/output/colbertv2.0", config=config)
        indexer.index(name="hotpotqa_wiki.nbits=2", collection="/root/paddlejob/workspace/env_run/output/ColBERT-main/psgs_w100.tsv",overwrite=True,)
        # indexer.index(name="hotpotqa_wiki.nbits=2", collection="small.tsv",overwrite=True)
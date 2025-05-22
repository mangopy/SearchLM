import json
import matplotlib.pyplot as plt
import numpy as np
from utilize.metrics import eval_acc, eval_f1, eval_em
from src.utilize.apis import  document_retrieval
from tqdm import tqdm
import ast

# file = './new_best_of/8/eval_data_wikimultihop_dev.searchagent_qwen7_2wiki1.inference.0.1400.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_musiqueqa_dev.searchagent_Qwen7_warmup1000.inference.0.140000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_nq_dev.checkpoint-300_.inference.0.140000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_nq_dev.agent0_checkpoint-240.inference.0.14000000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_nq_dev.agent_qwen14_checkpoint-560.inference.0.14000000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.agent1_test_checkpoint-330.inference.0.100000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_musiqueqa_dev.agent0_checkpoint-240.inference.0.1000000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_wikimultihop_dev.qwen7_2wiki3_.inference.0.1400.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_wikimultihop_dev.agent0_checkpoint-240.inference.0.1400.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_train.qwen7_align3_checkpoint-100.inference.0.40000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_Qwen7_warmup12000.inference.0.1000000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_qwen14_warmup4000.inference.0.1000000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_Qwen7_warmup2000.inference.0.14000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_qwen7_align3.inference.0.20000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_qwen7_align2.inference.0.20000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_qwen7_align1.inference.0.20000.json'

# 0.6363912621785095

pred = json.load(open(file))['output']

def compuate_golden():
    """/root/paddlejob/workspace/env_run/output/searchagent/hotpot-traj.100.json
    """
    ...
    

def searchLM(file, k=2):
    results = []
    bar = tqdm(pred)
    precision = 0
    recall = 0
    count = 0
    results = []
    for i, line in enumerate(bar, 1):
        if line['answer'] == 'yes' or line['answer'] == 'no':
            continue
        queries = []
        for e in line['messages']:
            if e['role']=='assistant' and '<QUERY>' in e['content'] and 'end search.' not in e['content']:
                queries.append(e['content'].replace('<QUERY>', ''))
        if queries == []:
            continue
        docs = document_retrieval(queries[-1], k=5)
        tmp = []
        for doc in docs:
            a =  eval_acc([doc], [[line['answer']]])['Acc'] 
            tmp.append(1 if a==100 else 0)
        if tmp == []:
            continue
        results.append(tmp)
        # precision += sum(tmp) / len(tmp)
        # recall += any(tmp)
        # count += 1
        # bar.set_postfix(precision=precision / count, recall = recall / count)

    print(file.split('/')[-1])
    for k in [1,3,5]:
        p = sum([sum(line[:k])/k for line in results]) / len(results)
        print(f"Precision@{k}: {p}")

    for k in [1,3,5]:
        r = sum([any(line[:k]) for line in results]) / len(results)
        print(f"Recall@{k}: {r}")


searchLM(file, k=3)
exit()

def colbert2(file):
    precision, recall = [], []
    results = []
    data = json.load(open(file))
    for line in tqdm(data):
        # docs = document_retrieval(line['question'], k=5)
        key = 'negative' if 'negative' in line else 'retrieval'
        docs = line[key][:5]
        tmp = []
        for doc in docs:
            a =  eval_acc([doc], [[line['answer']]])['Acc'] 
            tmp.append(1 if a==100 else 0)
        results.append(tmp)

    print(file.split('/')[-1])
    for k in [1,3,5]:
        p = sum([sum(line[:k])/k for line in results]) / len(results)
        print(f"Precision@{k}: {p}")

    for k in [1,3,5]:
        r = sum([any(line[:k]) for line in results]) / len(results)
        print(f"Recall@{k}: {r}")

# root = '/root/paddlejob/workspace/env_run/output/searchagent/data/eval_data'
# for e in ['hotpotqa_dev', 'musiqueqa_dev', 'nq_dev', 'wikimultihop_dev', ]: # 'hotpotqa_train', 'musiqueqa_train', 'nq_train', 'wikimultihop_train']:
    # colbert2(f"{root}/{e}.json")

def searchr1(file):
    data = json.load(open(file))
    precision, recall = [], []
    results = []
    data = json.load(open(file))
    for line in data:
        try:
            output = line['predict'].split('\n')
            output = [ast.literal_eval(e.replace('<information>', '').replace('</information>','')) for e in output if e.startswith('<information>')]
            docs = sum(output, [])[-5:]
            tmp = []
            for doc in docs:
                a =  eval_acc([doc], [[line['answer']]])['Acc'] 
                tmp.append(1 if a==100 else 0)
            results.append(tmp)
        except:
            pass        
    print(len(results))
    print(file.split('/')[-1])
    for k in [1,3,5]:
        p = sum([sum(line[:k])/k for line in results]) / len(results)
        print(f"Precision@{k}: {p}")

    for k in [1,3,5]:
        r = sum([any(line[:k]) for line in results]) / len(results)
        print(f"Recall@{k}: {r}")

searchr1('/root/paddlejob/workspace/env_run/output/Search-R1-main/wikimultihop_dev.json')
# colbert25
# MonoT5
# 

# 0.3872260150916275
# 0.6882860222781172

# def naive_colbert():
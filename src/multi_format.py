import json
from collections import defaultdict
import random

file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_Qwen7_warmup500_em1.inference.0.20000.json'
data = json.load(open(file))['output']

for line in data:
    cnt = [1 for e in line['messages'] if '<SEARCH>' in e['content'] and e['role']=='user']
    if sum(cnt)<=2:
        print('='*50)
        print('\n'.join([e['content'] for e in line['messages']]))
        print(f"answer: {line['answer']}")
import json
import random
from src.utilize.metrics import eval_acc, eval_f1, eval_em

files = [
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-musiqueqa_train.json.0.100000.SearchAgent_agent1_checkpoint-339.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-hotpotqa_train.json.0.60000.SearchAgent_agent0_checkpoint-240.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-nq_train.json.0.60000.SearchAgent_agent0_checkpoint-240.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-nq_train.json.0.60000.SearchAgent_agent3_hotpot_1_checkpoint-400.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-nq_train.json.0.60000.SearchAgent_agent2_nq_checkpoint-300.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-wikimultihop_train.json.0.60000.SearchAgent_agent0_checkpoint-240.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-wikimultihop_train.json.0.60000.SearchAgent_agent1_2wiki_checkpoint-200.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-wikimultihop_train.json.0.60000.SearchAgent_agent2_2wiki_checkpoint-400.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-musiqueqa_train.json.0.60000.SearchAgent_agent0_checkpoint-240.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-musiqueqa_train.json.0.60000.SearchAgent_agent1_musique_checkpoint-291.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/log/eval_data-musiqueqa_train.json.0.60000.SearchAgent_agent2_musique_checkpoint-300.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/best_of/8/eval_data-musiqueqa_train.json.0.60000.SearchAgent_agent_mistral-24_checkpoint-200.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/best_of/8/eval_data-musiqueqa_train.json.0.60000.SearchAgent_agent_llama13b_1_checkpoint-150.json'
    # '/root/paddlejob/workspace/env_run/output/SearchAgent/best_of/8/eval_data-wikimultihop_train.json.0.60000.SearchAgent_agent_mistral-24_checkpoint-200.json'
    '/new_best_of/8/eval_data_hotpotqa_train.searchagent_Qwen7_warmup1000.inference.0.20000.json'
]
add = False
is_rank = True
dataset = 'hotpotqa'
epo = 1
data = sum([json.load(open(file))['output'] for file in files], [])
print(eval_acc([line['pred'] for line in data], [[line['answer']] for line in data]))
print(eval_em([line['pred'] for line in data], [[line['answer']] for line in data]))
print(eval_f1([line['pred'] for line in data], [[line['answer']] for line in data]))
print(len(data))
random.shuffle(data)

results = []
cnt = 0
for line in data:
    r = eval_acc([line['pred']], [[line['answer']]])['Acc']
    # print(r)
    if r != 100.00:
        continue
    cnt += 1
    # if cnt>20000:
        # continue
    results.append({"messages": line['messages']})
    if is_rank:
        results += line['rank']

print(len(results))

if add:
    ref_data = json.load(open('/root/paddlejob/workspace/env_run/output/SearchAgent/hotpot-traj.all.json'))
    # ref_data = [{"messages": line['messages']} for line in ref_data]
    results += ref_data

random.shuffle(results)
print(len(results))
output_file = f'/root/paddlejob/workspace/env_run/output/SearchAgent/data/train/mistral24/{dataset}/epo{epo}/{dataset}.{len(results)}.json'
json.dump(results, open(output_file, 'w'), indent=4)
print(output_file)
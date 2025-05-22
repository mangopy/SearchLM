import json
import matplotlib.pyplot as plt
import numpy as np


def plot_count(data, dataset):
    # 假设有3年数据，共1095天
    days = list(range(1, len(data) + 1))
    y1 = [e[0] for e in data]
    y2 = [e[1] for e in data]

    # 创建图形
    plt.figure(figsize=(14, 6))

    # 绘制曲线和填充面积
    plt.plot(days, y1, label='predicted', color='orange')
    plt.fill_between(days, y1, color='orange', alpha=0.3)

    plt.plot(days, y2, label='answer', color='skyblue')
    plt.fill_between(days, y2, color='skyblue', alpha=0.3)

    tick_positions = np.linspace(1, len(data), num=6).tolist()
    tick_labels = [f'Query {i}' for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)

    # 图表设置
    plt.xlabel('Query ID')
    plt.ylabel('Search turns')
    plt.title(f'Search turns trend in across {len(data)} example in dataset')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'count-{dataset}6.png', dpi=300, bbox_inches='tight')
    plt.show()

# days = list(range(1, 3 * 365 + 1))
# jinan_temp = 10 + 10 * np.sin(np.linspace(0, 6*np.pi, len(days))) + np.random.randn(len(days))
# qingdao_temp = 12 + 8 * np.sin(np.linspace(0, 6*np.pi, len(days))) + np.random.randn(len(days))
# plot_count([[e1, e2] for e1, e2 in zip(jinan_temp, qingdao_temp)])

# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_nq_train.agent_qwen14_checkpoint-560.inference.0.140000.json'
file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_musiqueqa_dev.agent0_checkpoint-240.inference.0.1000000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_train.qwen7_align3_checkpoint-100.inference.0.40000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_Qwen7_warmup12000.inference.0.1000000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_qwen14_warmup4000.inference.0.1000000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_Qwen7_warmup2000.inference.0.14000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_qwen7_align3.inference.0.20000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_qwen7_align2.inference.0.20000.json'
# file = '/root/paddlejob/workspace/env_run/output/searchagent/new_best_of/8/eval_data_hotpotqa_dev.searchagent_qwen7_align1.inference.0.20000.json'


pred = json.load(open(file))['output']
pred = {line['query_id']: line for line in pred}

root = '/root/paddlejob/workspace/env_run/output/searchagent/data/eval_data'

for e in ['hotpotqa_dev', 'musiqueqa_dev', 'nq_dev', 'wikimultihop_dev', 'hotpotqa_train', 'musiqueqa_train', 'nq_train', 'wikimultihop_train']:
    if e in file:
        dataset = e
        data = json.load(open(f"{root}/{e}.json"))

def count(arr):
    cnt = 0
    for line in arr:
        if line['role'] == 'assistant': # and "<query>" in line['content'].lower():
            cnt += 1
    return cnt

results = []
for line in data:
    if line['query_id'] not in pred:
        continue
    p = count(pred[line['query_id']]['messages']) # + len(pred[line['query_id']]['rank'])
    a = len(set(line['positive']))
    # if p > 8:
        # continue
    results.append([p, a])
print(sum([e[-1] for e in results])/len(results))
print(sum([e[0] for e in results])/len(results))
print(len(results))
results = sorted(results, reverse=False, key=lambda x: x[0])
plot_count(results, dataset)
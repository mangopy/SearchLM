from src.instruction import *
import numpy as np
from src.utilize.utilize import *
from rouge import Rouge
import glob
from src.utilize.apis import *

def _eval_rouge(pred: str, answer: str, metric='rouge-l'):
    rouge = Rouge()
    return rouge.get_scores(pred, answer, avg=True)[metric]['f']

def hotpot_prepare(data):
    cleaner = TextNormalization()

    def _prepare(line):
        traj = line['traj'][-1]['content'].split('\n')
        traj = [t for t in traj if cleaner.is_search(t) or cleaner.is_reference(t) or cleaner.is_keyword(t) or cleaner.is_final(t)]
        prompt = PROMPT.format(task=line['question'])
        responses = ''
        context = {c[0]: c[1] for c in line['context']}
        refs = [f'[{i}] ' + context[fact[0]][fact[1]].strip() for i, fact in enumerate(line['supporting_facts'])]
        i = 0
        rank = []
        # document mapping
        while i < len(traj):
            # print('===', traj[i])
            if cleaner.is_final(traj[i]):
                break
            while i < len(traj) and not cleaner.is_search(traj[i]):
                i+=1

            query = cleaner.normalize(traj[i])
            i+=1

            index = []
            while i < len(traj) and cleaner.is_reference(traj[i]):
                index += extract_numbers_from_ordered_brackets(cleaner.normalize(traj[i]))
                i+=1
            _refs = [refs[idx] for idx in index]
            candidates = document_retrieval(query, k=20)
            # candidates = ['da3/envs/LlamaTuning/bin/python -X pycache_prefix=/Users/shizhl/Library/Caches/JetBrains/PyCharm2024.3/cpython-c'] * 20
            score = [[_eval_rouge(ref, doc, 'rouge-l') for doc in candidates] for ref in _refs]
            labels = np.argmax(np.array(score), axis=1).tolist()

            assert cleaner.is_keyword(traj[i])
            answer = cleaner.normalize(traj[i])

            responses += cleaner.formulate(
                idx=math.ceil(int(i/3)),
                query=query,
                doc=' '.join([f"[{label}] "+ candidates[label] for label in labels]),
                answer=answer
            )
            rank.append({"prompt": rank_instruction(question=query, candidates=candidates), "response": '>'.join([f'[{label}]' for label in labels])})
            i+=1

        responses = responses + FINAL_SENTENCE
        responses = responses.split('\n')
        i = 0
        messages = []
        while i < len(responses):
            response = []
            while not responses[i].startswith('<SEARCH>'):
                response.append(responses[i])
                i+=1
            messages.append({"role": "assistant", "content": '\n'.join(response)})
            messages.append({"role": "user", "content": responses[i]})
            i+=1

        messages = [{"role": "user", "content": prompt}] + messages + [{"role": "assistant", "content": line["answer"]}]

        return messages, rank

    results = []
    results_wo_rank = []
    for line in tqdm(data):
        try:
            messages, rank = _prepare(line)
            results.append({"messages": messages})
            results_wo_rank.append({"messages": messages})
            results += [{"messages": [{"role": "user", "content:": _rank['prompt']}, {"role": "assistant", "content": _rank['response']}]} for _rank in rank]
        except:
            print(line['traj'][-1]['content'])
            pass
    print(len(results) / len(data))
    return results, results_wo_rank



if __name__ == '__main__':
    files = glob.glob('/root/paddlejob/workspace/env_run/output/SearchAgent/data/warmup/hotpot/*.json')
    data = sum([json.load(open(file)) for file in files], [])
    results, results_wo_rank = hotpot_prepare(data)
    write_file(results, f'./hotpot-traj.all.{len(results)}.json')
    write_file(results_wo_rank, f'./hotpot-traj.wo-rank.{len(results_wo_rank)}.json')

"""
task
<QUERY> When was "Arthur's Magazine" first published?
<SEARCH> [0] da3/envs/LlamaTuning/bin/python -X pycache_prefix=/Users/shizhl/Library/Caches/JetBrains/PyCharm2024.3/cpython-c

<FACT> 1844
<QUERY> When was "First for Women" first published?
<SEARCH> [0] da3/envs/LlamaTuning/bin/python -X pycache_prefix=/Users/shizhl/Library/Caches/JetBrains/PyCharm2024.3/cpython-c

<FACT> 1989
<Final>
<SEARCH> end search. please give the final answer to the input question: {question} 

"""
import json
from tqdm import tqdm
import os
from src.utilize.utilize import *
from src.utilize.apis import get_from_openai


system = """You are an intelligent search agent, which can simulate the question-answering process based on my question and answer."""

Input = """Given an open-domain query about Wikipedia, I have already marked the correct answer at the end of the question and provided all the reference Wikipedia passages needed to answer the question.
Your task is to reformat my provided question and references into a detailed question-answering process.
Specifically, there should be three types of special tokens in your output:
1. <Search>, followed by a sub-query
2. <Reference>, followed by the citation ID
3. <keyword>, followed by the answer to the sub-query

Since this is a multi-hop question, your output should interleave the `<Search>`, `<Reference>`, and `<keyword>` tokens until you reach the final answer.
Please start with a special token `<Final>` followed by the final answer.

Here is a concrete example to demonstrate the output format:
```example
Question: Which magazine was started first, Arthur's Magazine or First for Women? (Answer: Arthur's Magazine)
Reference:
[1] Arthur's Magazine | Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured works by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846, it was merged into "Godey's Lady's Book".
[2] First for Women | First for Women is a women's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011, the circulation of the magazine was 1,310,696 copies.
Your Output:
<Search 1> When did the magazine "Arthur's Magazine" start?
<Reference 1> [1]
<Keyword 1> 1844
<Search 2> When did the magazine "First for Women" start?
<Reference 2> [2]
<Keyword 2> 1989
<Final> Arthur's Magazine
```

Starting below, for the question "{question}", please complete your output following the above requirements.

Question: {question}
Reference:
{ref}
Your Output:"""

def hotpot_traj(file, output_file,left, right):
    data = json.load(open(file))
    results = []
    for line in tqdm(data[left:right]):
        try:
            query = line['question'] + f'(Answer: {line["answer"]})'
            context = {c[0]: c[1] for c in line['context']}
            ref = [f'[{i}] ' + context[fact[0]][fact[1]].strip() for i, fact in enumerate(line['supporting_facts'])]
            prompt = Input.format(question=query, ref='\n'.join(ref))
            print(prompt)
            message = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            output = get_from_openai(model_name='gpt-4o', messages=message)
            line['gpt'] = output
            line['traj'] = message + [{'role': "assistant", "content": output['content']}]
            print(line['traj'])
            results.append(line)
        except:
            print(line)
    write_file(results, output_file)

def printf(file):
    data = load_data(file)
    for line in data:
        print()
        print('-'*20)
        query = line['question']
        context = {c[0]: c[1] for c in line['context']}
        ref = [f'[{i}] ' + context[fact[0]][fact[1]].strip() for i, fact in enumerate(line['supporting_facts'])]
        print(query, line['answer'])
        print('\n'.join(ref))
        print(line['traj'][-1]['content'])

left = 10000
right = 12000
output_file = f'hotpot.traj-gpt-4o.{left}.{right}.json'
hotpot_traj('/Users/shizhl/Paper2024/SearchAgent/data/raw/hotpotqa/hotpot_train_v1.1.json', output_file, left, right)
printf(output_file)

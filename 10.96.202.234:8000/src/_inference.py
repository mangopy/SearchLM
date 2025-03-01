import argparse
from email import message

from httpx import main
from src.utilize.metrics import eval_acc, eval_f1, eval_em
from src.utilize.utilize import *
from src.instruction import *
from vllm import LLM, SamplingParams
from src.utilize.apis import get_from_openai, document_retrieval
from src.utilize.utilize import printf

BEST_OF = os.environ.get('best_of', 8)
N = os.environ.get('N', 1)
printf(f"HYPER> Best of = {BEST_OF}, return N = {N}")

class SearchLM:
    def __init__(self, model_name, devices: List):
        self.llm = LLM(model=model_name, tensor_parallel_size=len(devices))

    def generation(self, inputs, stop=None, probs=False):
        response  = self.complete(prompt=inputs, stop=stop) if type(inputs) == str else self.chat(messages=inputs, stop=stop)
        return response

    def complete(self, prompt, stop=None):
        if stop is None:
            stop = []
        base_kwargs = {"top_p": 0.95, "max_tokens": 256, "temperature": 0.8, "stop": stop, "logprobs": 1, "best_of": BEST_OF, "n": N}
        sampling_params = SamplingParams(**base_kwargs)
        response = self.llm.generate(prompt, sampling_params=sampling_params)
        return response.outputs[0]

    def chat(self, messages, stop=None):
        if stop is None:
            stop = []
        base_kwargs = {"top_p": 0.95, "max_tokens": 256, "temperature": 0.8, "stop": stop, 'logprobs': 1, "best_of": BEST_OF, "n": N}
        sampling_params = SamplingParams(**base_kwargs)
        response = self.llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        return response[0].outputs[0]


def _inference(model: SearchLM, example, max_iter=8, k=20):
    prompt = PROMPT.format(task=example['question'])
    messages = [{"role": "user", "content": prompt}]
    rank = []
    cleaner = TextNormalization()
    flag = False
    for i in range(max_iter):
        response = model.generation(messages, stop=[SEARCH_TOKEN]).text
        if FINAL_TOKEN in response:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": FINAL_SENTENCE.format(question=example["question"])})
            flag = True
            break

        if i == 0:
            fact = None
            query = response
        else:
            try:
                fact, query = response.split(THINK_TOKEN)
            except:
                fact = response.split('\n')[0]
                fact = cleaner.normalize(fact)
                query = model.generation(inputs=messages + [{"role": "user", "content": f"{FACT_TOKEN} {fact}\n{THINK_TOKEN}"}], 
                                         stop=[SEARCH_TOKEN]).text

        fact = cleaner.normalize(fact)
        query = cleaner.normalize(query)
        print(query)
        candidates = document_retrieval(query, k=k)
        rank_input = rank_instruction(question=query, candidates=candidates)
        doc_ids = model.generation(inputs=[{"role": "user", "content": rank_input}], stop=[]).text

        labels = extract_numbers_from_ordered_brackets(cleaner.normalize(doc_ids))
        rank.append([{"role": "user", "content": rank_input}, {"role": "assistant", "content": '>'.join([f'[{label}]' for label in labels])}])

        print('doc ids: ', doc_ids)
        docs = get_doc_by_ids(doc_ids=doc_ids, candidates=candidates, max_id=k)
        docs = '\n'.join(docs)
        traj = f"{FACT_TOKEN} {fact}\n{THINK_TOKEN} {query}" if fact is not None else f"{THINK_TOKEN} {query}"
        print(traj)
        messages.append({"role": "assistant", "content": traj})
        messages.append({"role": "user", "content": f"{SEARCH_TOKEN} {labels} {docs}"})

    if not flag:
        messages[-1]['content'] += '\n' + FINAL_SENTENCE.format(question=example["question"])
    output = model.generation(inputs=messages)
    pred = cleaner.normalize(output.text)
    messages.append({"role": "assistant", "content": pred})
    print(example['question'], "ANSWER: " + example['answer'], "PRED: " + pred)
    return {"query_id": example['query_id'], 'answer': example['answer'], "pred": pred, "messages": messages, "rank": [{"messages": r} for r in rank],}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse configuration.")
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--input_file", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=False)

    args = parser.parse_args()

    data = load_data(args.input_file)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    devices = devices.split(',')
    sllm = SearchLM(args.model_name, devices)
    results = []
    cnt = 0
    for example in tqdm(data):
        try:
            result = _inference(model=sllm, example=example)
            results.append(result)
        except:
            cnt += 1
    print(cnt)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

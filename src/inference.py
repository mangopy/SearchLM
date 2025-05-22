import logging
import json
import os.path
import subprocess
import time
import math
import argparse
from httpx import main
from src.utilize.metrics import eval_acc, eval_f1, eval_em
from src.utilize.utilize import load_data, write_file, printf
from vllm import LLM, SamplingParams
import numpy as np
import random

logger = logging.getLogger('evaluation')

global_log = '/root/paddlejob/workspace/env_run/output/global_log'
# WARM_UP_FILE = '/root/paddlejob/workspace/env_run/output/searchagent/hotpot-traj.all.json'
WARM_UP_FILE = '/root/paddlejob/workspace/env_run/output/searchagent/hotpot-traj.all.11771.json'

class RAGEvaluator:
    @staticmethod
    def evaluate(preds, answers):
        score = {}
        score['f1'] = eval_f1(preds, [[e] if type(e) == str else e for e in answers])['F1']
        score['em'] = eval_em(preds, [[e] if type(e) == str else e for e in answers])['EM']
        score['acc'] = eval_acc(preds, [[e] if type(e) == str else e for e in answers])['Acc']
        print(score)
        return score


def paralle_launch(model_name, file, data, devices):
    processes = []
    length = math.floor(len(data) // len(devices)) + 1
    output_files = []
    for ids, device in enumerate(devices):
        # time.sleep(20)
        left, right = ids * length, (ids + 1) * length
        input_file = os.path.join(global_log, f'raw.{left}.{right}.json')
        write_file(data[left: right], input_file)

        output_file = os.path.join(global_log, f'cache.{left}.{right}.json')
        log_file = os.path.join(global_log, f'log.{left}.{right}.txt')

        # torchrun --master_addr {port} --nproc_per_node=1
        command = f"""CUDA_VISIBLE_DEVICES={device} python {file} \
--model_name {model_name} \
--input_file {input_file} \
--output_file {output_file} > {log_file} 2>&1"""

        process = subprocess.Popen(command, shell=True)
        processes.append(process)
        output_files.append(output_file)

    print(f'launch {len(processes)} processes...')
    for process in processes:
        process.wait()
    print(f'end {len(processes)} processes...')

    cache = sum([load_data(file) for file in output_files], [])

    # assert len(data) == len(cache)
    print(len(cache))
    return cache

    
# /root/paddlejob/workspace/env_run/output/SearchAgent/agent0/checkpoint-240
# /root/paddlejob/workspace/env_run/output/SearchAgent/agent1/checkpoint-339
# bamboogle, hotpotqa_train
# /root/paddlejob/workspace/env_run/output/SearchAgent/agent2_musique/checkpoint-300

def create_output_file(model_name, input_file, file_type, left, right):
    def parse_model_name(model_name):
        return '_'.join(model_name.split('/')[-2:])
    def parse_input_file(input_file):
        return '_'.join(input_file.split('/')[-2:]).replace('.json','')
    file = [
        parse_input_file(input_file), 
        parse_model_name(model_name),
        file_type, 
        f'{left}.{right}',
        'json'
    ]
    return '.'.join(file)


def inference():
    parser = argparse.ArgumentParser(description="Parse configuration.")
    parser.add_argument("--input_file", type=str, required=False, default="./data/eval_data/nq_train.json")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False, default='./log')
    parser.add_argument("--left", type=int, required=False, default=0)
    parser.add_argument("--right", type=int, required=False, default=60000)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    file = create_output_file(args.model_name_or_path, args.input_file, 'inference', args.left, args.right)
    output_file = os.path.join(args.output_dir, file)

    data = load_data(args.input_file)
    data = data[args.left:args.right]

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    results = paralle_launch(args.model_name_or_path, './src/_inference.py', data, devices)

    answers = [r['answer'] if 'answer' in r else r['answers'] for r in results]
    preds = [r['pred'] for r in results]
    score = RAGEvaluator.evaluate(preds=preds, answers=answers)
    with open(output_file, 'w') as f:
        json.dump({"score": score, "output": results, "config": vars(args)}, f, indent=4)
        printf(f"WRITING> finish writing the evaluation results on {output_file}...")


def entropy():
    parser = argparse.ArgumentParser(description="Parse configuration.")
    parser.add_argument("--inference_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False, default='./data/train')
    parser.add_argument("--left", type=int, required=False, default=0)
    parser.add_argument("--right", type=int, required=False, default=60000)
    parser.add_argument("--epo", type=int, required=False, default=1)
    parser.add_argument("--add_warmup", action="store_true", help="Enable warmup (default: False)")
    parser.add_argument("--is_rank", action="store_true", help="Enable warmup (default: False)")

    args = parser.parse_args()

    def count(messages):
        cnt = 0
        for line in messages:
            if line['role'] == 'assistant' and '<QUERY>' in line['content']:
                cnt+=1
        return cnt
    
    def fact(messages):
        cnt = 0
        for line in messages:
            if line['role'] == 'assistant' and ('<FACT> \n<QUERY>' in line['content'] or '<FACT> \n<QUERY> ' == line['content']):
                cnt+=1
        return cnt

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.inference_file)['output']
    data = data[args.left:args.right]

    for line in data:
        assert line['messages'][-1]['role'] == 'assistant'
        line['messages'][-1]['content'] = line['answer']

    if args.add_warmup:
        print('add the warm up data...')
        tmp = json.load(open(WARM_UP_FILE))
        data += tmp

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    print(len(data))
    # data = [line for line in data if count(line['messages']) <= 4]
    data = [line for line in data if fact(line['messages']) <= 2]
    print(len(data))
    entropy = paralle_launch(args.model_name_or_path, './src/_entropy.py', data, devices)
    for line1, line2 in zip(data, entropy):
        assert line1['query_id'] == line2['query_id']
    
    entropy_data_file = f'epo_{args.epo}.entropy.left={args.left}.right={min(args.right, len(entropy))}.json'
    write_file(entropy, os.path.join(args.output_dir, entropy_data_file))
    printf(f"WRITING> finish writing the entropy results on {entropy_data_file}...")

    # formuate
    results = []
    cnt = 0
    for line1, line2 in zip(data, entropy):
        assert line1['query_id'] == line2['query_id']
        # if 'answer' in line1 and 'pred' in line1 and eval_em([line1['answer']], [[line1['pred']]])['EM'] != 100:
            # continue
        prob = line2['log_probs']
        if math.isnan(prob):
            continue  
        results.append({"messages": line1['messages'], "prob": prob})
        cnt += 1
        if args.is_rank:
            results += [{"messages": e['messages'], "prob": prob} for e in line1['rank']]

    print(cnt, len(results))
    training_data_file = f'epo{args.epo}.training.len={len(results)}.json'
    write_file(results, os.path.join(args.output_dir, training_data_file))
    printf(f"WRITING> finish writing the training results on {training_data_file}...")

def reward():
    parser = argparse.ArgumentParser(description="Parse configuration.")
    parser.add_argument("--inference_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False, default='./data/train')
    parser.add_argument("--left", type=int, required=False, default=0)
    parser.add_argument("--right", type=int, required=False, default=60000)
    parser.add_argument("--epo", type=int, required=False, default=1)
    parser.add_argument("--metric", type=str, required=False, default='f1')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.inference_file)['output']
    data = data[args.left:args.right]

    results = []
    cnt = 0
    for line in data:
        answers = [line['answer']]
        preds = [line['pred']]
        score = RAGEvaluator.evaluate(preds=preds, answers=answers)
        results.append({"messages": line['messages'], "prob": score[args.metric]/100})
        cnt += score[args.metric]
    print(cnt/len(data))
    reward_data_file = f'epo_{args.epo}.{args.metric}.left={args.left}.right={min(args.right, len(results))}.json'
    write_file(results, os.path.join(args.output_dir, reward_data_file))
    printf(f"WRITING> finish writing the entropy results on {reward_data_file}...")


def formate():
    parser = argparse.ArgumentParser(description="Parse configuration.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--inference_file", type=str, required=True)
    parser.add_argument("--entropy_file", type=str, required=True)
    parser.add_argument("--epo", type=int, required=True)
    parser.add_argument("--size", type=int, required=False)
    parser.add_argument("--is_rank", type=bool, required=False, default=True)
    parser.add_argument("--add_warmup", type=bool, required=False, default=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.inference_file)['output']

    if args.add_warmup:
        print('add the warm up data...')
        tmp = json.load(open(WARM_UP_FILE))
        data += tmp

    entropy = load_data(args.entropy_file)
    results = []
    cnt = 0
    for line1, line2 in zip(data, entropy):
        # if eval_em([line1['answer']], [line1['pred']]) != 1.0:
            # continue
        assert line1['query_id'] == line2['query_id']
        if 'answer' in line1:
            line1['messages'][-1]['content'] = line1['answer'] 
        # prob = np.exp(-line2['log_probs']).item() 
        prob = line2['log_probs']
        results.append({"messages": line1['messages'], "prob": prob})
        cnt += 1
        if args.is_rank:
            results += [{"messages": e['messages'], "prob": prob} for e in line1['rank']]
    
    # random.shuffle(results)
    results = results[:args.size]

    write_file(results, os.path.join(args.output_dir, f'epo{args.epo}.training.len{len(results)}.json'))


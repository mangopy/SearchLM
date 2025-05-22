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

class RAGEvaluator:
    @staticmethod
    def evaluate(preds, answers):
        score = {}
        score['f1'] = eval_f1(preds, [[e] if type(e) == str else e for e in answers])
        score['em'] = eval_em(preds, [[e] if type(e) == str else e for e in answers])
        score['acc'] = eval_acc(preds, [[e] if type(e) == str else e for e in answers])
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


parser = argparse.ArgumentParser(description="Parse configuration.")
parser.add_argument("--output_file", type=str, required=False, 
                    default="/root/paddlejob/workspace/env_run/output/SearchAgent/data/eval_data/nq_train.json")
parser.add_argument("--size", type=int, required=False, default=1400)
args = parser.parse_args()
results = json.load(open(args.output_file))['output'][:args.size]
answers = [r['answer'] if 'answer' in r else r['answers'] for r in results]
preds = [r['pred'] for r in results]
score = RAGEvaluator.evaluate(preds=preds, answers=answers)

# R = F1 x P()
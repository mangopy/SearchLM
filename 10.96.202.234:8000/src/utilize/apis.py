import json
import sys
from openai import OpenAI
import time
import random
import requests

api_search = "http://10.96.202.234:8893"

def document_retrieval(query, k=20):
    url = f'{api_search}/api/search?query={query}&k={k}'
    response = requests.get(url)
    res = response.json()
    knowledge = []
    for doc in res['topk']:
        text = doc['text'][doc['text'].index('|') + 1:].replace('"','').strip()
        title = doc['text'][:doc['text'].index('|')].replace('"','').strip()
        knowledge.append(f"Title: {title}. Content: {text}")
    return knowledge


api_keys_list = [
    ('sk-LyjNjNUGmBs0xTftABbAq9m0WDDCQPKuLM6y3Y4tVHQMQCQK', 'https://api.chatanywhere.tech/v1'),
    ('sk-ScLGcKDUbasUIccuFa7uTRw3BSfvNzb22B4AtNfSyBmtpbAH', 'https://api.chatanywhere.tech/v1'),
]

def get_from_openai(model_name='gpt-3.5-turbo', messages=None, prompt=None, stop=None, max_len=1000, temp=1, n=1,
                    json_mode=False, usage=True):
    for i in range(10):
        try:
            key = random.randint(0, 100000) % len(api_keys_list)
            client = OpenAI(api_key=api_keys_list[key][0],
                            base_url=api_keys_list[key][1])
            kwargs = {
                "model": model_name, 'max_tokens': max_len, "temperature": temp,
                "n": n, 'stop': stop,
            }
            begin = time.time()
            if json_mode == True:
                kwargs['response_format'] = {"type": "json_object"}
            if 'instruct' in model_name and 'gpt' in model_name:
                # assert prompt != None or messages!=None
                kwargs['prompt'] = prompt if prompt != None else messages[0]['content']
                response = client.completions.create(**kwargs)
            else:
                assert messages is not None
                kwargs['messages'] = messages
                response = client.chat.completions.create(**kwargs)
            end = time.time()

            cost = end-begin
            content = response.choices[0].message.content if n == 1 else [res.message.content for res in response.choices]
            results = {"content": content}
            results['time'] = cost
            if usage == True:
                results['usage'] = [response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens]
            return results
        except:
            error = sys.exc_info()[0]
            print("API error:", error)
            time.sleep(120)
    return 'no response from openai model...'

# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType
)
from swift.utils import seed_everything

model_type = ModelType.qwen2_5_7b_instruct
template_type = 'qwen'

model_id_or_path = 'QLUNLP/BianCang-Qwen2.5-7B'
model, tokenizer = get_model_tokenizer(model_type, model_id_or_path=model_id_or_path, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256

template = get_template(template_type, tokenizer)
seed_everything(42)
query = '你好，你是谁？'  # Hello, who are you?
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

# ========================================================================

import json
with open('./data/dataset_origin.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

import random
nums = list(range(len(data)))
order = random.sample(nums, len(nums))
print(order)

''' This can also be used for verification tests by using "type". '''
def get_query(name, ingredients, type):
    if type == 0:
        # What are the ingredients of the drug {name}?
        # Only give a list of the names of the ingredients please.
        # No need to give dosages or the production workflow.
        return f'请给出药物：{name}的组成成分？只需给出成分名称的列表即可，不需要给出剂量或制备流程。'
    if type == 2:
        # Whether the drug {name} consists of {ingredients}? Only respond with "yes" or "no".
        return f'药物：{name}的组成成分是否为：{ingredients}？只需回答“是”或“否”。'

TP, FP, TN, FN = 0, 0, 0, 0
temp_his = []

ord = 0
for i in order:
    query = get_query(data[i]['name'], data[i]['ingredients'], 0)
    response, history = inference(model, template, query, history)
    ord += 1
    print(f'第{ord}个问题：')  # The {ord}-th question:
    print(f'ID: {i} query: {query}')
    print(f'ID: {i} response: {response}')
    answer_this = data[i]['ingredients']
    print(f'ID: {i} correct answer：{answer_this}')
    temp_his.append({'ID': i, 'query': query, 'response': response, 'med': data[i]['name'], 'ingr': data[i]['ingredients']})
    # # '是' is 'yes' and '否' is 'no'.
    # if response[0] == '是':
    #     if data[i]['answer'] == 1:
    #         TP += 1
    #     else:
    #         FP += 1
    # if response[0] == '否':
    #     if data[i]['answer'] == 0:
    #         TN += 1
    #     else:
    #         FN += 1

print(f'history: {history}')

num = 0
for i in temp_his:
    result = {
        "model": model_id_or_path,
        "order": num,
        "ID": i['ID'],
        "medicine_name": i['med'],
        "response": i['response'],
        "true_ingredients": i['ingr']
    }
    with open(f'./data/direct/{model_id_or_path}.json', 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        f.write('\n')
    num += 1

# ========================================================================

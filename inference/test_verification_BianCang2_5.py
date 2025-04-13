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
# dataset = "subset_true"
# dataset = "subset_false"
dataset = "dataset_origin"
with open(f'./data/{dataset}.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

import random
nums = list(range(len(data)))
order = random.sample(nums, len(nums))
print(order)

def get_query(name, ingredients):
    # Whether the drug {name} consists of {ingredients}? Only respond with "yes" or "no".
    return f'药物：{name}的组成成分是否为：{ingredients}？只需回答“是”或“否”。'

TP, FP, TN, FN = 0, 0, 0, 0

ord = 0
for i in order:
    query = get_query(data[i]['name'], data[i]['ingredients'])
    response, history = inference(model, template, query, history)
    ord += 1
    print(f'第{ord}个问题：')  # The {ord}-th question:
    print(f'ID: {i} query: {query}')
    print(f'ID: {i} response: {response}')

    # '是' is 'yes' and '否' is 'no'.
    if response[0] == '是':
        if data[i]['answer'] == 1:
            TP += 1
        else:
            FP += 1
    if response[0] == '否':
        if data[i]['answer'] == 0:
            TN += 1
        else:
            FN += 1

print(f'history: {history}')

accu, prec, recl, fone = 0, 0, 0, 0

if len(data) == 0:
    accu = 0
else:
    accu = (TP+TN)/len(data)

if TP+FP == 0:
    prec = 0
else:
    prec = TP/(TP+FP)

if TP+FN == 0:
    recl = 0
else:
    recl = TP/(TP+FN)

if 2*TP+FP+FN == 0:
    fone = 0
else:
    fone = 2*TP/(2*TP+FP+FN)

print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')
print(f'accuracy: {TP+TN}/len(data) = {accu}')
print(f'precision: {prec}')
print(f'recall: {recl}')
print(f'F1: {fone}')

result = {
    "dataset": dataset,
    "model": model_id_or_path,
    "case_number": len(data),
    "TP": TP,
    "FP": FP,
    "TN": TN,
    "FN": FN,
    "accuracy": accu,
    "precision": prec,
    "recall": recl,
    "f1": fone
}
with open('./data/verification/results.json', 'a', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
    f.write('\n')

# ========================================================================

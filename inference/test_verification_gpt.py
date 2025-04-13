import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

# role = 'user'
# role = 'assistant'
role = 'system'
query = 'You are a traditional Chinese medical scientist. 我们将用简体中文对话。'  # We will talk in simplified Chinese.
model_id_or_path = 'gpt-3.5-turbo'
temperature = 0

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": role,
            "content": query,
        }
    ],
    # prompt = query,
    model=model_id_or_path,
    temperature=temperature
)

print(f'query: {query}')
print(f'response: {chat_completion.choices[0].message.content}')

# =================================================
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
    # You are a seasoned trditional Chinese medicine expert.
    # Whether the drug {name} consists of {ingredients}? Only respond with "yes" or "no".
    return f'你是一名资深中医专家。请问药物：{name}的组成成分是否为：{ingredients}？只需回答“是”或“否”。'

TP, FP, TN, FN = 0, 0, 0, 0
history = []

ord = 0
for i in order:
    query = get_query(data[i]['name'], data[i]['ingredients'])
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": role,
                "content": query,
                }
            ],
        # prompt = query,
        model=model_id_or_path,
        temperature=temperature
        )
    response = chat_completion.choices[0].message.content
    history.append({'ID': i, 'query': query, 'response': response})
    ord += 1
    print(f'第{ord}个问题：')
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

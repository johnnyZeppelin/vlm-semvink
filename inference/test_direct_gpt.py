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
with open('./data/dataset_origin.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

import random
nums = list(range( len(data) ))
order = random.sample(nums, len(nums))
print(order)

''' This can also be used for verification tests by using "type". '''
def get_query(name, ingredients, type):
    if type == 0:
        # You are a seasoned trditional Chinese medicine expert.
        # What are the ingredients of the drug {name}?
        # Only give a list of the names of the ingredients please.
        # No need to give dosages or the production workflow.
        # ask with only name is type 0; with only ingredients is type 1; with name and ingredients is type 2.
        return f'你是一名资深中医专家。请给出药物：{name}的组成成分？只需给出成分名称的列表即可，不需要给出剂量或制备流程。'
    if type == 2:
        # You are a seasoned trditional Chinese medicine expert.
        # Whether the drug {name} consists of {ingredients}? Only respond with "yes" or "no".
        return f'你是一名资深中医专家。请问药物：{name}的组成成分是否为：{ingredients}？只需回答“是”或“否”。'

TP, FP, TN, FN = 0, 0, 0, 0
history = []

ord = 0
for i in order:
    query = get_query(data[i]['name'], data[i]['ingredients'], 0)
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
    history.append({'ID': i, 'query': query, 'response': response, 'med': data[i]['name'], 'ingr': data[i]['ingredients']})
    ord += 1
    print(f'第{ord}个问题：')  # The {ord}-th question:
    print(f'ID: {i} query: {query}')
    print(f'ID: {i} response: {response}')
    ans = data[i]['ingredients']
    print(f'ID: {i} correct answer: {ans}')
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
for i in history:
    result = {
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

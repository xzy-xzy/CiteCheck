from config_api import config
import json
from tqdm import tqdm

f = open(f"../dataset/{config.aim}.jsonl").readlines( )
samples = [ ]
for i, x in tqdm(enumerate(f)):
    x = json.loads(x)
    s = x["statement"]
    q = x["quote"]
    text = f"判断陈述是否完全得到参考文本的支撑。陈述：{s} 参考文本：{q} 答案（仅输出一个字，是或否）："
    samples.append({"input": text, "label": x["label"]})

label2id = {"是": 1, "否": 0}
tot_c, rit_c = [0, 0], [0, 0]

# exit(0)

from openai import OpenAI
from time import sleep

max_length = 2

if config.base_url is None:
    client = OpenAI(api_key=config.api_key)
else:
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

def get_response(text):
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": text}],
            max_tokens=max_length,
            temperature=0,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return None


import os
from tqdm import tqdm

root = f"../archive_{config.model}/{config.aim}"

if not os.path.exists(root):
    os.makedirs(root)

# exit(0)

i = -1

for x in tqdm(samples):
    i += 1
    if os.path.exists(f"{root}/{i}.txt"):
        continue
    text = x["input"]
    output = get_response(text)
    if output is None:
        continue
    o = open(f"{root}/{i}.txt", "w")
    o.write(output)
    o.close( )
    # sleep(1.5)

for i, x in enumerate(samples):
    o = open(f"{root}/{i}.txt").read( )
    gold = x["label"]
    for z in o:
        if z in label2id:
            pred = label2id[z]
            break
    else:
        pred = -1
    tot_c[gold] += 1
    rit_c[gold] += (pred == gold)

acc = sum(rit_c) / sum(tot_c) * 100
acc_c = [x / y * 100 for x, y in zip(rit_c, tot_c)]

print(acc, acc_c)





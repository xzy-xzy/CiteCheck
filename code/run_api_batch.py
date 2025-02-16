from config_api import config
import json
from pathlib import Path
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
import os

max_length = 2

if config.base_url is None:
    client = OpenAI(api_key=config.api_key)
else:
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

name = f"{config.model}_{config.aim}"
root = f"../archive_{config.model}/{config.aim}"

if not os.path.exists(root):
    os.makedirs(root)

if not os.path.exists(f"{root}/id.txt"):

    f = open(f"{root}/batch.jsonl", "w")

    for idx, x in enumerate(samples):
        text = x["input"]
        body = {
            "model": config.model,
            "max_tokens": max_length,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": text}
            ]
        }
        batch = {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        f.write(json.dumps(batch) + "\n")

    f.close( )

    if "qwen" in config.model:
        batch_input_file = client.files.create(
          file=Path(f"{root}/batch.jsonl"),
          purpose="batch"
        )

    elif "gpt" in config.model:
        batch_input_file = client.files.create(
          file=open(f"{root}/batch.jsonl", "rb"),
          purpose="batch"
        )

    else:
        print("Unknown model")
        batch_input_file = client.files.create(
          file=open(f"{root}/batch.jsonl", "rb"),
          purpose="batch"
        )

    f = open(f"{root}/id.txt", "w")
    f.write(batch_input_file.id)
    f.close( )

    print("Batch created.")

else:
    print("Batch already created.")

idx = open(f"{root}/id.txt").read( )

if not os.path.exists(f"{root}/bid.txt"):
    ret = client.batches.create(
        input_file_id=idx,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": f"{name}"
        }
    )
    bid = ret.id
    f = open(f"{root}/bid.txt", "w")
    f.write(bid)
    f.close( )
    print("Batch submitted.")

else:
    bid = open(f"{root}/bid.txt").read( )
    print("Batch already submitted.")


if not os.path.exists(f"{root}/result"):
    ret = client.batches.retrieve(bid)
    # print(ret)
    if ret.status == "completed":
        os.mkdir(f"{root}/result")
        idx = ret.output_file_id
        response = client.files.content(idx)
        text = response.text
        # f = open(f"{root}/result.txt", "w")
        text = text.split("\n")
        if len(text[-1]) == 0:
            text = text[:-1]
        ret = [ ]
        for line in text:
            t = json.loads(line)
            custom_id = int(t["custom_id"])
            content = t["response"]["body"]["choices"][0]["message"]["content"]
            ret.append((custom_id, content))
        ret.sort(key=lambda x: x[0])
        for idx, content in ret:
            f = open(f"{root}/result/{idx}.txt", "w")
            f.write(content)
            f.close( )
        print("Batch completed.")
    else:
        print("Batch not completed.")
        exit(0)

else:
    print("Batch already completed.")


for i, x in enumerate(samples):
    gold = x["label"]
    o = open(f"{root}/result/{i}.txt").read( )
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



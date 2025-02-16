import gc
import os

import sys
import threading

import numpy as np
import psutil
import torch
import copy
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from scheduler import get_scheduler
from transformers import default_data_collator
from typing import Dict, Optional, Sequence

from peft import LoraConfig, TaskType, get_peft_model
from collections import Counter

import yaml
from config_tune import config


def main():
    accelerator = Accelerator()
    location = config.loc
    model_name = config.model
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=8, lora_alpha=32,
                             lora_dropout=0.1)

    lr = 5e-4
    num_epochs = 10
    batch_size = 256
    batch_size_per_gpu = 2
    warmup_ratio = 0.03
    scheduler_type = "cosine"
    seed = 42
    train_max_length = 2048
    save_mode = True
    save_steps = 3000
    eval_epochs = 1
    corpus_dir = "../dataset"
    saved_name = f"../archive_{model_name.replace('/', '_')}/" + \
                 f"_LoRA_epoch{num_epochs}_lr{lr}_batch{batch_size}_" \
                 f"{scheduler_type}_warmup{warmup_ratio}_seed{seed}"
    set_seed(seed)

    # determine gradient accumulation steps
    f = open("../myconfig_lora.yaml", "r")
    acc_config = yaml.load(f, Loader=yaml.FullLoader)
    accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = batch_size
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = batch_size_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = \
        batch_size // batch_size_per_gpu // acc_config["num_processes"]

    print(accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"])


    id2label = {0: "否", 1: "是"}
    label2id = {"否": 0, "是": 1}


    data_dict = {
        "train": "train.jsonl"
    }
    dataset = load_dataset(path=corpus_dir, data_files=data_dict)

    print("num_rows: ", dataset["train"].num_rows)

    tokenizer = AutoTokenizer.from_pretrained(location + model_name)
    S = set([id2label[x["label"]] for x in dataset["train"]])
    for x in S:
        print(x, tokenizer(x)["input_ids"])

    s1, s2 = tokenizer("B")["input_ids"], tokenizer("C")["input_ids"]
    cut_st = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            cut_st = i
            break

    print("cut_st:", cut_st)

    # exit(0)
    target_max_length = max([len(tokenizer(id2label[x["label"]])["input_ids"][cut_st:]) for x in dataset["train"]])
    print("target_max_length: ", target_max_length)

    prefix = "判断陈述是否完全得到参考文本的支撑。 "

    label_column = "label"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    uncut_str = " 答案（是或否） ： "
    uncut_str_length = len(uncut_str)
    uncut = tokenizer(uncut_str)["input_ids"]
    print(uncut)
    uncut = uncut[cut_st:]
    print(uncut)
    uncut_length = len(uncut)

    def preprocess_function(example):
        statement = example["statement"]
        reference = example["quote"]
        source = prefix + " 陈述 ： " + statement + " 参考文本 ： " + reference + uncut_str
        target = id2label[example["label"]]
        src_id = tokenizer(source, max_length=train_max_length, truncation=True)["input_ids"]
        src_length = len(src_id)
        tgt_id = tokenizer(target, truncation=True)["input_ids"][cut_st:]
        src_tgt_id = src_id + tgt_id + [tokenizer.eos_token_id]
        labels = [-100] * src_length + src_tgt_id[src_length:]
        src_tgt_id = src_tgt_id[:train_max_length]
        labels = labels[:train_max_length]
        label_id = example["label"]
        return {"input_ids": src_tgt_id, "labels": labels, "exceed": src_length == train_max_length,
                "label_id": label_id}

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=False,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    print("Training Dataset Done.")

    accelerator.wait_for_everyone( )

    train_dataset = processed_datasets["train"]
    train_dataset = [x for x in train_dataset if not x["exceed"]]

    max_length = max([len(x["input_ids"]) for x in train_dataset])
    print("max_length: ", max_length)

    print(len(train_dataset))
    # exit(0)

    eval_datasets = [ ]
    part_datasets = [ ]

    for name in ["dev.jsonl", "test.jsonl"]:

        data_dict = {
            "test": name
        }
        dataset = load_dataset(path=corpus_dir, data_files=data_dict)

        def test_preprocess_function(example):
            statement = example["statement"]
            reference = example["quote"]
            source = prefix + " 陈述 ： " + statement + "参考文本 ： " + reference
            input_ids = tokenizer(source)["input_ids"]
            input_ids += uncut
            # input_ids = torch.tensor(input_ids)
            return {"input_ids": input_ids, } # "label": example["label"]}

        with accelerator.main_process_first():
            processed_datasets = dataset.map(
                test_preprocess_function,
                batched=False,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )

        eval_datasets.append(processed_datasets["test"])
        part_datasets.append(dataset["test"])
        print(len(processed_datasets["test"]))

    print("Eval Dataset Done.")

    accelerator.wait_for_everyone( )

    def padding(batch, padding_value):
        max_len = max([len(x) for x in batch])
        return [[padding_value] * (max_len - len(x)) + x for x in batch]    # Left-padding

    def train_collate_fn(instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in instances]
        labels = [x["labels"] for x in instances]
        input_ids = padding(input_ids, tokenizer.pad_token_id)
        labels = padding(labels, -100)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        label_ids = [x["label_id"] for x in instances]
        return dict(
            input_ids=input_ids,
            labels=labels,
            label_ids=label_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    def test_collate_fn(instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in instances]
        input_ids = padding(input_ids, tokenizer.pad_token_id)
        input_ids = torch.tensor(input_ids)
        # labels = [x["label"] for x in instances]
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            # labels=labels
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=train_collate_fn, batch_size=batch_size_per_gpu, pin_memory=True
    )
    eval_dataloaders = [
        DataLoader(
            x, collate_fn=test_collate_fn, batch_size=batch_size_per_gpu, pin_memory=True
        ) for x in eval_datasets
    ]
    # test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)

    # creating model
    model = AutoModelForCausalLM.from_pretrained(location + model_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_scheduler(scheduler_type, train_dataloader, num_epochs, warmup_ratio, optimizer)

    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )
    eval_dataloaders = [accelerator.prepare(x) for x in eval_dataloaders]
    print(eval_dataloaders)
    accelerator.print(model)

    eval_names = ["dev", "test"]
    eval_parts = part_datasets

    if accelerator.is_main_process:
        if not os.path.exists(f"./{saved_name}"):
            os.makedirs(f"./{saved_name}")
        if not os.path.exists(f"./{saved_name}/eval_res"):
            os.makedirs(f"./{saved_name}/eval_res")
        if not os.path.exists(f"./{saved_name}/model"):
            os.makedirs(f"./{saved_name}/model")


    if os.path.exists(f"./{saved_name}/state/epoch_record.txt"):
        f = open(f"./{saved_name}/state/epoch_record.txt", "r")
        epoch_begin = int(f.readline( ).strip( )) + 1
        accelerator.load_state(f"./{saved_name}/state")
    else:
        epoch_begin = 0

    print(f"start from epoch {epoch_begin}")

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    def evaluate(step_name):

        model.eval( )

        print(f"Evaluation {step_name} starts.")

        for eval_dataloader, eval_name, part_dataset in zip(eval_dataloaders, eval_names, eval_parts):

            print(f"\n\n{eval_name}\n\n")
            eval_preds = [ ]

            for _, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                input_length = batch["input_ids"].shape[1]
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3, max_new_tokens=target_max_length,
                        pad_token_id=tokenizer.eos_token_id
                    )  # synced_gpus=True for DS-stage 3
                    outputs = outputs[:, input_length:]

                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs)

                preds = preds.detach().cpu().numpy()
                results = tokenizer.batch_decode(preds, skip_special_tokens=True)
                eval_preds.extend(results)

            accelerator.wait_for_everyone( )

            eval_preds = [label2id[x] if x in label2id else x for x in eval_preds]
            gold_labels = part_dataset["label"]
            idxs = part_dataset["idx"]

            assert len(eval_preds) == len(gold_labels)

            tot_c, rit_c = [0, 0], [0, 0]

            for x, y in zip(eval_preds, gold_labels):
                tot_c[y] += 1
                rit_c[y] += (x == y)

            acc = sum(rit_c) / sum(tot_c) * 100
            acc_c = [x / y * 100 for x, y in zip(rit_c, tot_c)]

            if accelerator.is_main_process:

                accelerator.print(acc, acc_c)
                f = open(f"./{saved_name}/eval_res/{eval_name}_{step_name}.txt", "w")
                for idx, x, y in zip(idxs, eval_preds, gold_labels):
                    f.write(f"{idx}\t{x}\t{y}\n")
                f.write(f"{acc} {acc_c}")
                g = open(f"./{saved_name}/eval_res/{eval_name}.txt", "a")
                g.write(f"{step_name} {acc} {acc_c}\n")


        if accelerator.is_main_process:

            if save_mode:
                now = accelerator.get_state_dict(model)
                if now is not None:
                    # model
                    now = {k: v for k, v in now.items() if "lora" in k}
                    torch.save(now, f"./{saved_name}/model/{step_name}.pt")

        model.train( )


    for epoch in range(epoch_begin, num_epochs):

        print(f"Epoch {epoch} starts.")

        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # print(batch)
            label_ids = batch["label_ids"]
            batch = {k: v for k, v in batch.items() if k != "label_ids"}
            # print(batch["input_ids"].shape)
            outputs = model(**batch)
            loss = outputs.loss
            # print(label_ids, [len(x) for x in batch["input_ids"]], loss)
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print("total_loss:", total_loss)

        if (epoch + 1) % eval_epochs == 0:
            evaluate(f"epoch{epoch}")


    accelerator.wait_for_everyone( )
    print("This is a process.")


if __name__ == "__main__":
    main()

# CiteCheck
Dataset and code for the paper [CiteCheck: Towards Accurate Citation Faithfulness Detection](https://arxiv.org/abs/2502.10881).

## Dataset
The dataset is available as `*.jsonl` files in `dataset`. The key-value pairs included in the file are:
- `idx`: `int`. The index number of the sample.
- `query`: `str`. Questions input to the RAG system.
- `answer`: `str`. The answer given by the RAG system.
- `statement`: `str`. A citation-marked statement taken from the answer.
- `quote`: `str`. The cited documents retrieved by the RAG system (indicated by the statement's citation marks).
- `label`: `int`. The value `0` indicates a negative sample and `1` indicates a positive sample.
- `method`: `str`. The value `none` indicates the original sample, `ch` indicates the augmented sample with information changed, and `del` indicates the augmented sample with information deleted.

Complete LLM Augmentation traces are available at [this link](https://drive.google.com/file/d/1tqssEs_o1VZTabXKfnwilHvp1td0x9O3/view?usp=share_link).

## Code
Start with `cd code`. 

For zero-shot experiments on LLMs:
```
python3 run_api.py --model [model] --api_key [api_key]
```
You can specify the base_url for platforms other than OpenAI with `--base_url [base_url]` and test on the dev set with `--aim dev`. For LLMs that support the batch API, you can use `run_api_batch.py` instead of `run_api.py`.

For fine-tuning experiments on models (from HuggingFace):
```
accelerate launch --config_file config_lora.yaml fine-tuning.py --model [model]
```
You can also specify a local model. For example, if the model is stored in `../cache/Llama-3.1-8B-Instruct`:
```
accelerate launch --config_file config_lora.yaml fine-tuning.py --model Llama-3.1-8B-Instruct --loc ../cache/
```

## Citation
```
@misc{xu2025citecheckaccuratecitationfaithfulness,
      title={CiteCheck: Towards Accurate Citation Faithfulness Detection}, 
      author={Ziyao Xu and Shaohang Wei and Zhuoheng Han and Jing Jin and Zhe Yang and Xiaoguang Li and Haochen Tan and Zhijiang Guo and Houfeng Wang},
      year={2025},
      eprint={2502.10881},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.10881}, 
}
```



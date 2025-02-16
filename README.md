# CiteCheck
CiteCheck: Towards Accurate Citation Faithfulness Detection

## Dataset
The dataset is available as `*.jsonl` files in `dataset`. The key-value pairs included in the file are:
- `idx`: `int`. The index number of the sample.
- `query`: `str`. Questions input to the RAG system.
- `answer`: `str`. The answer given by the RAG system.
- `statement`: `str`. A citation-marked statement taken from the answer.
- `quote`: `str`. The cited documents (indicated by the statement's citation marks).
- `label`: `int`. The value `0` indicates a negative sample and `1` indicates a positive sample.
- `method`: `str`. The value `none` indicates the original sample, `ch` indicates the augmented sample with information changed, and `del` indicates the augmented sample with information deleted.

Complete LLM Augmentation traces are available at [this link](https://drive.google.com/file/d/1tqssEs_o1VZTabXKfnwilHvp1td0x9O3/view?usp=share_link).


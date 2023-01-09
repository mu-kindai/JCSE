# Introduction
This repository contains the data, code and models for our paper JCSE: Contrastive Learning of Japanese Sentence Embeddings and Its Applications. It is built upon Pytorch and Huggingface.

## Overview
We propose a novel Japanese sentence representation framework, JCSE for domain adaptation(derived from "Contrastive learning of Sentence Embeddings for Japanese"), that creates training data by generating sentences and synthesizing them with sentences available in a target domain. Specifically, a pre-trained data generator is finetuned to a target domain using our collected corpus. It is then used to generate contradictory sentence pairs that are used in contrastive learning with a two-stage training recipe for adapting a Japanese language model to a specific task in the target domain.

![overall image](/figure_overview.png)

## Requirements
We recommend the following dependencies.
+ Python 3.8
+ Pytorch 1.9
+ transformers 4.22.2
+ datasets 2.3.2
+ spaCy 3.3.1
+ GiNZA 5.1.2

## Japanese STS benchmark
Another problem of Japanese sentence representation learning is the difficulty of evaluating existing embedding methods due to the lack of benchmark datasets. Thus, we establish a comprehensive Japanese Semantic Textual Similarity (STS) benchmark on which various embedding models are evaluated.

We use the SentEval toolkit to evaluate embedding models on the Japanese STS benchmark which has been combined in our published [SentEval toolkit](/SentEval/data/downstream).

### Evaluation
You can evaluate any Japanese sentence embedding models following the commands below:
```
python evaluation.py \
    --model_name_or_path <your_model_dir> \
    --pooler  <cls|cls_before_pooler|avg|avg_top2|avg_first_last> \
    --task_set <sts|transfer|full> \
    --mode test
```

### Model List
The Japanese sentence embedding models trained by us are listed as following.
| Model |
| ----- |
| [MU-Kindai/Japanese-SimCSE-BERT-base-unsup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-base-unsup) |
| [MU-Kindai/Japanese-SimCSE-BERT-large-unsup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-large-unsup) |
| [MU-Kindai/Japanese-SimCSE-RoBERTa-base-unsup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-RoBERTa-base-unsup) |
| [MU-Kindai/Japanese-MixCSE-BERT-base](https://huggingface.co/MU-Kindai/Japanese-MixCSE-BERT-base) |
| [MU-Kindai/Japanese-MixCSE-BERT-large](https://huggingface.co/MU-Kindai/Japanese-MixCSE-BERT-large) |
| [MU-Kindai/Japanese-DiffCSE-BERT-base](https://huggingface.co/MU-Kindai/Japanese-DiffCSE-BERT-base) |
| [MU-Kindai/SBERT-JSNLI-base](https://huggingface.co/MU-Kindai/SBERT-JSNLI-base) |
| [MU-Kindai/SBERT-JSNLI-large](https://huggingface.co/MU-Kindai/SBERT-JSNLI-large) |
| [MU-Kindai/Japanese-SimCSE-BERT-base-sup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-base-sup) |
| [MU-Kindai/Japanese-SimCSE-BERT-large-sup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-large-sup) |

## JCSE
### Download data
Wikipedia data and JSNLI data for contrastive learning can be downloaded from [here](/data/download_wiki.sh) and [here](/data/download_nli.sh):
```
wget https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/wiki1m.txt
wget https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/nli_for_simcse.csv 
```



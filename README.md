# x-parser
a Chinese constituent parser

# Introduction
This is a project of the Chinese constituent parser with a novel variant of LSTM. Experiments on the ctb5 dataset show that the parser can achieve comparable performance with the state-of-the-art method in the F1-measure(91.5). Specially, the bert related code in the `./my_bert` is from [HuggingFace](https://github.com/huggingface/transformers).

# Usage
## Requirements
- `python 3.6.3`
- `torch 1.3.1`
- `numpy 1.17.2`

## Bert model
- L=12/H=768 (BERT-Base). You can download it [here](https://github.com/google-research/bert).
- Please unzip it and put it under `./data/pretrained_model`.

## Word embedding
- The word embedding in our experiment is the same as the work of [Zhang et al. (2018)](https://arxiv.org/abs/1805.02023). You can get it [here](https://github.com/jiesutd/LatticeLSTM).

## Data
- The [ctb5](https://catalog.ldc.upenn.edu/LDC2005T01) dataset is used. The unbinarized-tree files and binarized-tree files are respectively named as `*9.txt` and `*10.txt`.
- The naming of other files is self-explanatory, and the format has been given.
- The `test9_stanford.txt` is the result of Stanford POS Tagger with gold segmentation.
- In addition, you need to add `NULL` label in the two files of `cst`. 

## Train
- `python train_main.py`

## Output
- Please create a new folder named `output`. The checkpoint and other results is outputed to it.

## Test
- `python test_main.py`

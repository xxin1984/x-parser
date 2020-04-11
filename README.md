# x-parser
a Chinese constituent parser

# Introduction
This is a project of the Chinese constituent parser with a novel variant of LSTM. Experiments on the ctb5 dataset show that the parser can achieve comparable performance with the state-of-the-art method in the F1-measure. Specially, the bert related code is from [HuggingFace](https://github.com/huggingface/transformers).

# Usage
## Requirements
- `python 3.6.3`
- `torch 1.3.1`
- `numpy 1.17.2`

## Bert model
- L=12/H=768 (BERT-Base). You can download from [here](https://github.com/google-research/bert).

## Train
- `python train_main.py`

## Output
- The checkpoint and other results is outputed to `./output`.

## Test
- `python test_main.py`

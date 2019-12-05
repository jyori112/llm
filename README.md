# Locally Linear Mapping

Locally linear mapping projects word embeddings of an embedding space (source space) into other embedding space (target space) assuming that there are shared vocabularies among the two embedding spaces.

## Setup

Clone this repository

```
$ git clone https://github.com/jyori112/llm
$ cd llm
```

Install python packages

```
$ pip install -r requirements.txt
```

## Usage

Suppose that `[SRC_EMB]` is the embedding file for the source embedding space and `[TRG_EMB]` is the one for the target embedding space.
(there are shared vocabularies)

Then you can apply locally linear mapping with 10 neighbors to the `[INPUT]` file by

```
$ python -m llm [SRC_EMB] [TRG_EMB] --num-neighbors 10 < [INPUT]
```
The resulting embedding are printed on the `stdout`.

## Citation

If you use this code for research, please cite the following paper,

```
@inproceedings{sakuma2019conll,
    author    = {Sakuma, Jin and Yoshinaga, Naoki},
    title     = {Multilingual Model Using Cross-Task Embedding Projection},
    booktitle = {Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)},
    year      = {2019},
    pages     = {22--32}
}
```

# README
Dialogue generation automatic evaluation metric used by ACL 2020 Paper "[Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness](https://aclanthology.org/2020.acl-main.515/)".

## Available Metric
- BLEU-N
  - [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- Distinct-N
  - [A Diversity-Promoting Objective Function for Neural Conversation Models](https://aclanthology.org/N16-1014/)
- Entropy
  - [Sequence to Backward and Forward Sequences: A Content-Introducing Approach to Generative Short-Text Conversation](https://aclanthology.org/C16-1316/)

## Evaluation
```
$ python evaluation.py \
        --reference_file examples/reference.txt \
        --hypothesis_file examples/hypothesis.txt \
        --train_corpus_file examples/train.tsv \
        --subword_token Ġ

> bleu-1 -> 28.27467233044941
> bleu-2 -> 19.01298617089923
> bleu-3 -> 13.851299856516366
> bleu-4 -> 10.35376730261862
> distinct-1 -> 69.23076923076923
> distinct-2 -> 78.26086956521739
> entropy -> 6.245005017964289
```
> **Note**<br>
> - reference_file:  <br>
>    - Golden response file <br>
> - hypothesis_file: <br>
>    - Response file generated by the model <br>
> - train_corpus_file: <br>
>    - Training data file for measuring Entropy (query \t response). <br>
> - subword_token: <br>
>    - Tokenizer subword <br>

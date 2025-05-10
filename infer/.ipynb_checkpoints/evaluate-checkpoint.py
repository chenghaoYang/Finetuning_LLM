# evaluate.py
# -*- coding: utf-8 -*-
# @Time    : 2025/04/18 10:00

import numpy as np
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def evaluate(data):
    """
    data: List[Dict], 每项：
      {
        "text": "<模型生成文本>",
        "ref" : ["<参考答案1>", ...]
      }
    返回：
      {
        "bleu_score"   : float,
        "rouge-l_score": float,
        "bert_f1"      : float,
      }
    """
    # 1) BLEU
    bleu_scorer = BLEU()
    bleu_scores = []
    for d in data:
        bleu = bleu_scorer.sentence_score(
            hypothesis=d["text"],
            references=d["ref"],
        )
        bleu_scores.append(bleu.score)
    bleu_mean = float(np.mean(bleu_scores)) if bleu_scores else 0.0

    # 2) ROUGE‑L via rouge_score
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=True
    )
    rouge_scores = []
    for d in data:
        # 取第一个参考
        ref = d["ref"][0]
        hyp = d["text"]
        sc = rouge_scorer_obj.score(ref, hyp)
        rouge_scores.append(sc["rougeL"].fmeasure * 100)  # convert to percentage
    rouge_mean = float(np.mean(rouge_scores)) if rouge_scores else 0.0

    # 3) BERTScore
    hyps = [d["text"] for d in data]
    refs = [d["ref"][0] for d in data]
    if hyps:
        P, R, F1 = bert_score(
            cands=hyps,
            refs=refs,
            lang="en",               # 或 "zh"、"ja" 按你的语言
            rescale_with_baseline=True
        )
        bert_f1_mean = float(F1.mean()) * 100
    else:
        bert_f1_mean = 0.0

    return {
        "bleu_score"   : bleu_mean,
        "rouge-l_score": rouge_mean,
        "bert_f1"      : bert_f1_mean,
    }


if __name__ == "__main__":
    sample = [
        {
            "text": "to make people trustworthy you need to trust them",
            "ref" : ["the way to make people trustworthy is to trust them"]
        }
    ]
    metrics = evaluate(sample)
    print("BLEU   :", metrics["bleu_score"])
    print("ROUGE‑L:", metrics["rouge-l_score"])
    print("BERT‑F1:", metrics["bert_f1"])
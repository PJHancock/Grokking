# Grokking

Code accompanying [*The Effects of Online Batch Selection on Grokking*](https://drive.google.com/file/d/1Jd27JEVY3jPxOSXhQLjMi-KYFXuaFXe3/view?usp=drive_link) (Preston Hancock, April 2026).

## What is grokking?

Grokking is a surprising training dynamic in which a neural network first memorizes its training
data — reaching near-perfect training accuracy while validation accuracy sits near chance — and
then, after a much longer stretch of continued optimization, suddenly generalizes and validation
accuracy jumps to match training accuracy. First reported by [Power et al. (2022)](https://arxiv.org/abs/2201.02177)
on modular arithmetic tasks, it challenges the usual assumption that generalization follows quickly
from memorization.

[Liu et al. (2022)](https://arxiv.org/abs/2205.10343) later explained this as the emergence of
structured internal representations: the model's embeddings shift from an unstructured cluster
during memorization to a geometry that reflects the algebraic structure of the task once it
generalizes. [Lee et al. (2024)](https://arxiv.org/abs/2405.20233) built on that slow/fast-gradient
picture to propose **GrokFast**, which amplifies the slow-varying, low-frequency component of the
gradient responsible for representation learning, accelerating grokking by roughly an order of
magnitude.

## This project

This repo reproduces and extends those results, and asks a further question: does *online batch
selection* — choosing which examples to train on each step, rather than sampling uniformly —
change grokking dynamics, with or without GrokFast? We evaluate `FULL`, `UNIFORM`,
`RHOLOSS` ([Mindermann et al., 2022](https://arxiv.org/abs/2206.07137)), `DIVBS`
([Hong et al., 2024](https://arxiv.org/abs/2406.04872)), and `BAYESIAN` batch selection strategies
against a decoder-only Transformer on modular arithmetic and an MLP on MNIST, each with and
without GrokFast-EMA.

**Headline finding:** batch selection strategy makes little difference to when or how well grokking
occurs (DivBS gives a marginal speedup at best); GrokFast dominates, cutting time-to-generalization
by roughly an order of magnitude and improving final validation accuracy across the board. See the
paper for full results and the Representation Quality Index analysis.

## Repository layout

| Path | Description |
| --- | --- |
| [`grokking_algorithmic/`](grokking_algorithmic) | Decoder-only Transformer on modular arithmetic (`ModSubtractDataset`, p=96), adapted from [Sea-Snell/grokking](https://github.com/Sea-Snell/grokking), the original replication of Power et al.'s setup. |
| [`grokking_mnist/`](grokking_mnist) | MLP-on-MNIST grokking experiments, batch selection comparisons, and t-SNE/RQI representation analysis notebooks. |
| [`grokfast/`](grokfast) | Vendored reference implementation of [GrokFast](https://github.com/ironjr/grokfast) ([Lee et al., 2024](https://arxiv.org/abs/2405.20233)), used as the acceleration method throughout. |
| [`data/`](data) | Shared datasets. |

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## References

1. Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, [arXiv:2201.02177](https://arxiv.org/abs/2201.02177), 2022.
2. Liu et al., *Towards Understanding Grokking: An Effective Theory of Representation Learning*, [arXiv:2205.10343](https://arxiv.org/abs/2205.10343), NeurIPS 2022.
3. Lee et al., *GrokFast: Accelerated Grokking by Amplifying Slow Gradients*, [arXiv:2405.20233](https://arxiv.org/abs/2405.20233), 2024.
4. Mindermann et al., *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt*, [arXiv:2206.07137](https://arxiv.org/abs/2206.07137), ICML 2022.
5. Hong et al., *Diversified Batch Selection for Training Acceleration*, [arXiv:2406.04872](https://arxiv.org/abs/2406.04872), ICML 2024.
6. Yuntian Deng, Alexander M. Rush, and Graham Neubig. *Bayesian Data Selection for Data-Efficient Training*, arXiv preprint arXiv:2203.09635, 2022.

# marimo-comp: Neural Cellular Automata as Computational Priors for LLMs

[![marimo](https://img.shields.io/badge/Run%20with-marimo-yellow)](https://marimo.io)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive reproduction and extension of [**Training Language Models via Neural Cellular Automata**](https://arxiv.org/abs/2603.10055) (Lee et al., MIT, 2026).

### Executive Summary
Pre-training LLMs on natural language is costly and finite. This project explores the hypothesis that **computational structure**, not semantics, drives LLM reasoning. By pre-pre-training on synthetic data from Neural Cellular Automata (NCA), we achieve:
- **10x Data Efficiency:** 164M NCA tokens outperform 1.6B natural language tokens.
- **1.6x Faster Convergence:** Accelerated downstream language modeling.
- **6% Lower Perplexity:** Consistent gains across text, math, and code.

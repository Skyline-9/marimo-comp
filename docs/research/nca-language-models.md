# Training Language Models via Neural Cellular Automata

**Paper:** Lee, Han, Kumar, Agrawal (MIT, 2026) — [arXiv:2603.10055](https://arxiv.org/abs/2603.10055)
**Competition pick: #1 recommendation for WASM notebook**

---

## 1. Core Finding

Pre-pre-training a 1.6B Llama on 164M tokens of NCA-generated data (2D grid trajectories) improves downstream language modeling by 5-7% perplexity and accelerates convergence 1.6x — outperforming pre-pre-training on the same amount of natural language (C4).

**Key claim:** Structure, not semantics, drives the emergence of reasoning in LLMs. NCA data provides a "purer" computational signal than natural language.

## 2. The NCA Architecture

- 2D grid: 12x12, periodic boundaries, 10-state alphabet
- Update rule: 3x3 convolution (4 channels) → cell-wise MLP (hidden=16, ReLU) → 10 logits per cell
- Mild stochasticity: softmax temperature τ=10⁻³
- For each sequence: random θ (network weights) AND random initial grid
- Tokenization: 2x2 patches → vocabulary of 10⁴ tokens, row-major serialization

## 3. Complexity Control

Gzip compression ratio as proxy for Kolmogorov complexity:
- r = compressed_bytes / raw_bytes × 100
- Only retain NCAs with r > 50% (less compressible = more complex)
- **Critical finding:** Optimal complexity is domain-dependent
  - OpenWebText/OpenWebMath: best transfer from 50%+ gzip band
  - CodeParrot: best from 30-40% gzip band (code is inherently less complex)

## 4. Results

| Metric | NCA Pre-pre-train | C4 Pre-pre-train (same tokens) | From scratch |
|--------|-------------------|--------------------------------|--------------|
| OWT Perplexity | 5.7% better | 3.2% better | baseline |
| Convergence | 1.6x faster | 1.2x faster | baseline |
| GSM8K pass@1 | 4.4% | 3.9% | 3.8% |
| HumanEval pass@1 | 7.5% | 6.5% | 6.8% |

**NCA with 164M tokens beats C4 with 1.6B tokens.** 10x less data, better results.

## 5. Transfer Mechanism

- **Attention layers carry the transferable priors** (re-initializing attention hurts most)
- MLPs encode domain-specific patterns (less transferable)
- Embedding layers are re-initialized for language vocabulary

## 6. Why This is Perfect for WASM Competition

1. **NCA is a for-loop over a grid** — pure numpy, runs in milliseconds
2. **Incredibly visual** — animated grids evolving, emergent patterns
3. **Controllable complexity** — slider for gzip threshold, alphabet size
4. **Natural extensions** — compare Wolfram rules, measure complexity, connect to "Edge of Chaos"
5. **GitHub repo exists:** [danihyunlee/nca-pre-pretraining](https://github.com/danihyunlee/nca-pre-pretraining)

## 7. Extension Papers

| # | Paper | Key Idea | WASM Demo |
|---|-------|----------|-----------|
| 1 | Intelligence at the Edge of Chaos (2410.02536, ICLR 2025) | ECA rule complexity → LLM capability; sweet spot at "edge of chaos" | Run all 256 ECA rules, measure complexity, plot vs downstream perf |
| 2 | NCA for ARC-AGI (2506.15746) | NCA applied to abstract reasoning grids | Solve ARC-style grid puzzles with NCA in browser |
| 3 | DiffLogic Cellular Automata (2506.04912) | Logic-gate NCA = interpretable rules | Compare logic vs continuous NCA |
| 4 | Universal pre-training by iterated random computation (2506.20057) | Theoretical justification for random/CA pretraining | Visualize why random computation produces learnable structure |
| 5 | Transformers on Procedural Data (2505.22308) | Modular structures emerge from procedural data | Show attention patterns on CA sequences |
| 6 | CA, many-valued logic, and deep NNs (2404.05259) | Theory of learning CA rules | Interactive rule complexity taxonomy |
| 7 | LifeGPT (2409.12182) | GPT trained on Game of Life | Interactive Game of Life predictor |
| 8 | Emergence of sparse attention (2505.17863) | How data distribution shapes attention | Visualize attention under different CA rules |
| 9 | CellARC (2511.07908) | Benchmark for reasoning using 1D CA | Run CellARC tasks in browser |
| 10 | Epiplexity (2601.03220) | Measure learnable information in CA data | Compute epiplexity of different CA rules — connects to Paper #2! |

## 8. Notebook Plan

### Section 1: What is a Cellular Automaton?
- Interactive: pick a Wolfram rule (0-255), watch 1D ECA evolve
- Slider for grid width, number of steps
- Color-coded by Wolfram class (I-IV)

### Section 2: From 1D to 2D — Neural Cellular Automata
- Interactive: random NCA with controllable parameters
- Show how random neural network weights → diverse dynamics
- Animate the 2D grid evolution

### Section 3: Complexity Matters
- Compute gzip compression ratio for each NCA trajectory
- Interactive histogram: filter by complexity band
- Show the "sweet spot" — too simple (periodic) or too random (incompressible) both useless

### Section 4: The Paper's Key Result
- Pre-computed results from the paper (bar charts, scaling curves)
- Interactive: compare NCA vs C4 vs Dyck pre-pre-training

### Section 5: Edge of Chaos (Extension)
- Run all 256 ECA rules, compute Lempel-Ziv complexity
- Plot complexity vs "learnability" (how well a small model predicts next state)
- Show the edge of chaos: Class IV rules (Rule 110, 54) are the sweet spot

### Section 6: Epiplexity Connection (Extension)
- Compute approximate epiplexity of different CA rules
- Show that Rule 54 (complex) has high epiplexity, Rule 30 (chaotic) has low

### Section 7: Your Turn — Design an NCA
- Interactive: user adjusts NCA parameters (kernel size, hidden dim, alphabet)
- See how it affects complexity and visual output
- Challenge: can you create a rule that's maximally complex but still structured?

## 9. Technical Constraints (WASM)
- numpy: grid operations, convolutions (scipy.signal.convolve2d or manual)
- matplotlib: animations via frame sequences, heatmaps
- No threading (single-threaded WASM)
- 2GB memory limit — 12x12 grids with 10 states are tiny
- All pre-computed LLM results embedded as dicts/JSON

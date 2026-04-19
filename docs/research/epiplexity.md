# From Entropy to Epiplexity: Rethinking Information for Bounded Intelligence

**Paper:** Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson (CMU + NYU, 2026) — [arXiv:2601.03220](https://arxiv.org/abs/2601.03220)
**Competition pick: #2 (has pip package, novel concept, good but less visual than NCA)**

---

## 1. Core Finding

Classical information theory (Shannon entropy, Kolmogorov complexity) assumes unlimited computation. For computationally bounded observers (like neural networks), information behaves differently:
- Deterministic transformations CAN create information
- Data ordering DOES matter
- Models CAN extract more structure than exists in the generator

Epiplexity formalizes this: it's the structural, learnable information content of data for a bounded observer.

## 2. The Three Paradoxes Resolved

### Paradox 1: "Information can't be created"
Classical: H(f(X)) ≤ H(X) for deterministic f.
Reality: A CSPRNG turns a short seed into apparently random output. For a bounded observer, this IS new information.
Epiplexity: Formally proves time-bounded entropy CAN increase under deterministic transformations.

### Paradox 2: "Order doesn't matter"
Classical: H(X,Y) = H(X|Y) + H(Y) regardless of factorization order.
Reality: LLMs learn chess better from moves→board than board→moves.
Epiplexity: The "harder" direction (board→moves) forces higher epiplexity extraction.

### Paradox 3: "Likelihood = distribution matching"
Classical: The true generator is the perfect model.
Reality: Conway's Game of Life has trivial local rules but emergent global patterns. A bounded observer needs a MORE complex model than the generator.
Epiplexity: Formalizes "emergence" — when the observer's model exceeds the generator's complexity.

## 3. The Framework

Total description = |Program| + E[-log P(data)]
- **Epiplexity S_T(X):** Program length (structural information extracted)
- **Time-bounded entropy H_T(X):** Residual randomness after structure extraction
- Subject to compute budget T

### Estimation Methods
1. **Prequential coding (heuristic):** Area under loss curve during training
2. **Requential coding (rigorous):** Compress student model using teacher checkpoints

## 4. Key Empirical Results

### Elementary Cellular Automata
| Rule | Behavior | Epiplexity | Time-bounded Entropy |
|------|----------|------------|---------------------|
| 15 | Simple patterns | Low | Low |
| 30 | Chaotic | Low | High |
| 54 | Complex/emergent | High | High |

Rule 54 is the sweet spot: generates both randomness AND learnable structure.

### Chess
- Board→moves ordering: higher epiplexity, better OOD generalization
- Moves→board: lower epiplexity, worse OOD transfer

### Natural Data (at 6e18 FLOPs)
OpenWebText > Chess > CIFAR-5M in epiplexity
This explains why language pretraining transfers best to diverse tasks.

## 5. Connection to NCA Paper (#7)

The NCA paper (2603.10055) cites epiplexity as theoretical motivation! The connection:
- NCA generates data with controllable complexity
- Epiplexity measures whether that complexity is LEARNABLE
- Rule 54 (high epiplexity) → good pretraining signal
- Rule 30 (low epiplexity, high entropy) → bad pretraining signal
- This explains WHY the NCA paper's gzip filter works: it selects for epiplexity

## 6. Extension Papers

| # | Paper | Key Idea |
|---|-------|----------|
| 1 | V-information (2002.10689) | Usable information under computational constraints (predecessor) |
| 2 | DataRater (2505.17895) | Meta-learned dataset curation |
| 3 | Data Pruning for LLMs (2309.04564) | When less data is more |
| 4 | Entropy-based Data Pruning (2406.14124) | Information entropy for sample importance |
| 5 | Neural Entropy (2409.03817) | Diffusion models and information theory |
| 6 | Laplace Sample Information (2505.15303) | Bayesian sample informativeness |
| 7 | Binarized NNs and Compression (2505.20646) | Learning-as-compression hypothesis |
| 8 | Measuring Info Transfer in NNs (2009.07624) | Estimating Kolmogorov complexity in NNs |
| 9 | Invariance and Disentanglement (1706.01350) | Information minimality in deep representations |
| 10 | Intelligence at Edge of Chaos (2410.02536) | CA complexity shapes learning — measure with epiplexity! |

## 7. Why #2 Not #1 for Competition

Pros:
- Has a [pip package](https://pypi.org/project/epiplexity/) (pure Python, might work in WASM)
- Novel concept that's intellectually exciting
- Natural interactive demos (compute epiplexity of user-chosen data)
- Connects beautifully to NCA paper

Cons:
- The concept is abstract — harder to make visually "wow"
- Computing epiplexity properly requires training neural networks (slow in WASM)
- Simpler approximations (gzip, Lempel-Ziv) lose the nuance
- Less "bring to life" potential than animated NCA grids

## 8. Best Use: As Extension IN the NCA Notebook

Rather than a standalone notebook, epiplexity works best as a section in the NCA notebook:
- "Why does complexity matter? Let's measure epiplexity"
- Compare CA rules by epiplexity using simple compression-based approximation
- Show the correlation: high epiplexity CA rules → better pretraining signal
- This gives the NCA notebook theoretical depth judges will love

## 9. WASM-Feasible Demos

1. Compute gzip/LZ complexity of user-generated sequences
2. Show the epiplexity decomposition (structure vs noise) on toy data
3. Compare ECA rules by approximate epiplexity
4. Interactive: generate data, see how epiplexity changes with transformations
5. The "paradox demos": show how shuffling reduces learnability (Paradox 2)

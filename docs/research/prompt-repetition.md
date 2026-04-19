# Prompt Repetition in Large Language Models

A research synthesis on how repeating input prompts improves LLM performance, the underlying mechanism rooted in causal attention limitations, and the broader landscape of repetition-based techniques.

---

## 1. The Core Finding

Leviathan, Kalman, and Matias (Google Research, December 2025) demonstrated that transforming a user query from `<QUERY>` to `<QUERY><QUERY>` consistently improves LLM accuracy for non-reasoning tasks across all tested models and benchmarks, without increasing output length or inference latency.

**Results summary:**
- 7 models tested: Gemini 2.0 Flash/Flash Lite, GPT-4o/4o-mini, Claude 3 Haiku/3.7 Sonnet, DeepSeek V3
- 7 benchmarks: ARC Challenge, OpenBookQA, GSM8K, MMLU-Pro, MATH, NameIndex, MiddleMatch
- 47 out of 70 benchmark-model combinations showed statistically significant improvement (McNemar test, p < 0.1)
- 0 statistically significant losses
- Custom retrieval tasks showed dramatic gains: NameIndex accuracy on Gemini Flash Lite went from 21.33% to 97.33%

**Efficiency claim:** Repetition only extends the input (processed in the parallelizable prefill stage), not the output. Generated responses maintain the same format and length as baseline prompts.

**Variants tested:**
| Method | Template |
|--------|----------|
| Baseline | `<QUERY>` |
| Prompt Repetition | `<QUERY><QUERY>` |
| Verbose | `<QUERY> Let me repeat that: <QUERY>` |
| x3 | `<QUERY> Let me repeat that: <QUERY> Let me repeat that one more time: <QUERY>` |
| Padding (control) | `<QUERY> Ignore these periods: ........` |

The Padding control confirmed that gains come from semantic repetition, not increased input length.

**Source:** [arxiv.org/abs/2512.14982](https://arxiv.org/abs/2512.14982)

---

## 2. Why It Works: The Causal Attention Bottleneck

### The fundamental constraint

Decoder-only LLMs (GPT, Gemini, Claude, LLaMA, etc.) use causal attention: each token can only attend to tokens that precede it in the sequence. This means for a query like:

```
What is the capital of the country that borders Thailand to the east?
```

The token "What" cannot attend to "Thailand" or "east." The model must process "What" with zero knowledge of what's being asked about. By the time it reaches "east," it has full context but the early tokens' representations are already committed.

### How repetition fixes this

When the prompt is repeated:

```
What is the capital of the country that borders Thailand to the east?
What is the capital of the country that borders Thailand to the east?
```

Every token in the second occurrence can attend to every token in the first occurrence. The second "What" now has full context from the entire first pass, including "Thailand" and "east." This effectively provides bidirectional attention without any architectural change.

### Attention heatmap evidence

Xu et al. (2024) provided direct evidence via attention heatmaps on LLaMA-2: tokens in the second pass of a repeated question show strong attention to first-pass tokens that would otherwise be "future" tokens in a single pass. The attention pattern reveals that the model treats the first pass as context and the second pass as the "real" processing.

### Connection to Prefix LM

This mechanism is functionally similar to Prefix LM (Raffel et al., 2023), where a prefix region uses bidirectional attention while generation remains causal. Prompt repetition achieves an approximation of this without architectural modification -- the first copy of the prompt serves as the "prefix" that all second-copy tokens can fully attend to.

---

## 3. The Related Research Landscape

### 3.1 RE2: Re-Reading Improves Reasoning

**Authors:** Xu et al. (Microsoft, Chinese Academy of Sciences, UTS, Beihang, 2024)
**Paper:** [arxiv.org/abs/2309.06275](https://arxiv.org/abs/2309.06275)

RE2 adds an explicit instruction: `"Read the question again: <Q>"` after the initial question. Unlike raw prompt repetition, RE2 specifically targets reasoning tasks and works as a plug-in module compatible with Chain-of-Thought, Plan-and-Solve, and PAL.

**Key findings:**
- Consistent improvements across 14 reasoning benchmarks (arithmetic, commonsense, symbolic)
- Works on both instruction-tuned (ChatGPT) and base models (LLaMA-2 13B/70B)
- Optimal at 2-3 re-reads; further repetition degrades performance
- Perplexity analysis: re-reading decreases perplexity of generating the question but increases perplexity of generating the answer after the optimal number of reads
- The phrasing "Read the question again:" outperforms simply repeating `"Q: <question>"`

**Relationship to Prompt Repetition:** RE2 focuses on reasoning with explicit instruction; Prompt Repetition focuses on non-reasoning with raw duplication. They are complementary -- RE2's gains are neutral-to-positive when combined with CoT, while Prompt Repetition is neutral when reasoning is enabled.

### 3.2 Echo Embeddings: Repetition Improves Language Model Embeddings

**Authors:** Springer, Kotha, Fried, Neubig, Raghunathan (Carnegie Mellon, ICLR 2025)
**Paper:** [arxiv.org/abs/2402.15449](https://arxiv.org/abs/2402.15449)

Echo Embeddings applies the same insight to text embeddings. The prompt format is:

```
Rewrite the sentence: <x>; rewritten sentence: <x>
```

Embeddings are extracted only from the second occurrence of `<x>`.

**Key findings:**
- In zero-shot setting, matches LLM2Vec-unsupervised (which requires architectural modification + MLM training)
- Outperforms BERT-large and RoBERTa-large (fine-tuned bidirectional models)
- Mean pooling over second-occurrence tokens is critical; last-token pooling alone is insufficient
- Low sensitivity to exact prompt wording
- The mechanism is confirmed via synthetic experiments: classical causal embeddings fail when discriminative information appears late in the sentence, while echo embeddings succeed

**Core insight:** Autoregressive LLMs do not need bidirectional attention modification to produce high-quality embeddings. Repetition is sufficient.

### 3.3 Asking Again and Again: The Null Result

**Authors:** Shaier, Sanz-Guerrero, von der Wense (Colorado/Mainz, 2024)
**Paper:** [arxiv.org/abs/2412.07923](https://arxiv.org/abs/2412.07923)

This paper tested repeating only the question (not the full prompt) 1x, 3x, and 5x within reading comprehension prompts.

**Key findings:**
- No statistically significant improvement across any model or dataset (Friedman test p = 0.70)
- Tested on GPT-4o-mini, DeepSeek-V3, LLaMA-3.1, Mistral 7B, Phi-4
- Evaluated on SQuAD, HotPotQA, Natural Questions
- Paraphrasing the question (instead of exact repetition) sometimes decreased performance for larger models

**The critical distinction:** Repeating just the question provides no benefit. The full context must be repeated for gains to materialize. This makes sense mechanistically: the bottleneck is that early context tokens can't attend to later question tokens (and vice versa). Repeating only the question doesn't help because the context tokens still lack access to the question's later tokens.

### 3.4 Context Repetition (CoRe) for Multi-Hop Reasoning

**Authors:** Park et al. (Seoul National University, 2024)
**Paper:** [arxiv.org/abs/2410.07103](https://arxiv.org/abs/2410.07103)

CoRe addresses the "misordered context problem" in multi-hop QA by repeating the entire context k times (where k = number of supporting documents).

**Key findings:**
- Up to 30 percentage point F1 gains on 2WikiMultihopQA with LLaMA-3.1-8B
- Up to 70 percentage point accuracy gains on synthetic tasks
- 1-2 repetitions typically sufficient; diminishing returns after that
- Gains are largest when the initial document order is suboptimal
- Attention analysis: CoRe helps distribute attention more evenly across documents, mitigating the "lost-in-the-middle" problem
- Performance trajectories show initially poor orderings benefit most from repetition

**Mechanism:** By repeating context k times, the model encounters documents in multiple implicit orderings within its attention window. This increases the probability of encountering an effective reasoning path regardless of initial document arrangement.

### 3.5 Same Task, More Tokens: The Length Penalty

**Authors:** Levy, Jacoby, Goldberg (Bar-Ilan/AI2, 2024)
**Paper:** [arxiv.org/abs/2402.14848](https://arxiv.org/abs/2402.14848)

This paper introduces FLenQA, a controlled benchmark isolating the effect of input length on reasoning.

**Key findings:**
- Reasoning performance degrades significantly as input length increases, even at just 3000 tokens
- Degradation occurs far below models' advertised context windows
- Next-word prediction accuracy negatively correlates with reasoning accuracy (perplexity is not a reliable proxy)
- Chain-of-Thought does NOT mitigate length-induced degradation
- Failure modes include: refusal to answer, label bias toward "False," answering before reasoning, and failure to incorporate relevant facts in CoT steps
- Even "duplicate padding" (repeating relevant info) can degrade performance in some models

**Implication for prompt repetition:** There exists a fundamental tension. Repetition helps by providing bidirectional context, but it also doubles input length, which independently hurts reasoning. The net effect depends on which force dominates for a given task and model. For short queries (where repetition is cheap), the bidirectional benefit likely wins. For long contexts, the length penalty may dominate.

### 3.6 Selective Attention Improves Transformer

**Authors:** Leviathan, Kalman, Matias (Google Research, ICLR 2025)
**Paper:** [arxiv.org/abs/2410.02703](https://arxiv.org/abs/2410.02703)

By the same authors as Prompt Repetition. Selective Attention allows tokens to "mask" (forget) previous tokens that are no longer relevant.

**Key findings:**
- Parameter-free modification: reuses one attention head to compute a masking matrix
- Achieves equivalent performance to models with 2x the attention parameters
- Enables up to 47x KV-cache memory reduction while matching baseline quality
- Surpasses H2O, TOVA, and other pruning methods on quality-cost tradeoff
- Different layers learn different masking patterns (some sparse, some dense)

**Connection to Prompt Repetition:** These are two sides of the same coin. Prompt Repetition is the inference-time hack: give the model redundant context so it can attend to everything. Selective Attention is the architectural solution: let the model intelligently forget irrelevant context. Both address the same underlying problem -- causal attention forces the model to carry all past context, even when irrelevant, while preventing access to future context that may be critical.

### 3.7 Interpreting the Repeated Token Phenomenon

**Authors:** Yona, Shumailov, Hayes, Barbero, Gandelsman (Google DeepMind / Oxford / Berkeley, 2025)
**Paper:** [arxiv.org/abs/2503.08908](https://arxiv.org/abs/2503.08908)

This paper explains why LLMs fail to accurately repeat a single token and links it to the "attention sink" mechanism.

**Key findings:**
- The first attention layer identifies and "marks" the first token in a sequence (creating an attention sink)
- When a token is repeated many times, the first attention layer cannot distinguish the first token from the repeated sequence
- This causes abnormally high attention weights on repeated tokens, leading to divergent behavior
- The vulnerability extends beyond exact repetition to similar tokens ("cluster attack")
- A targeted correction (ablating specific "sink neurons") mitigates the issue without impacting general performance

**Implication for prompt repetition:** There is a potential failure mode when repetition is excessive. The attention sink mechanism can be disrupted by long identical sequences. This aligns with RE2's finding that more than 2-3 repetitions degrades performance. Prompt Repetition's approach of 1 additional copy appears to stay safely below this threshold.

### 3.8 Sequence Repetition for Sequence Labeling

**Paper:** [arxiv.org/abs/2601.07894](https://arxiv.org/abs/2601.07894) (January 2026)

Extends the repetition idea to sequence labeling tasks (NER, POS tagging) with decoder-only LLMs.

**Key finding:** Second-pass tokens receive richer representations from attending to the first pass, confirming the mechanism across yet another task family.

---

## 4. A Hierarchy of Repetition Strategies

The research collectively reveals a clear hierarchy of what works and what doesn't:

| Strategy | Effect | Explanation |
|----------|--------|-------------|
| Repeat just the question | No improvement | Context tokens still can't see question tokens (Shaier 2024) |
| Repeat full prompt (non-reasoning) | Consistent improvement | Full bidirectional context achieved (Leviathan 2025) |
| Re-read with instruction (reasoning) | Consistent improvement | Explicit instruction + bidirectional context (Xu et al. 2024) |
| Repeat context for multi-hop QA | Large improvement | Multiple implicit orderings overcome misordered docs (CoRe 2024) |
| Repeat for embeddings | Matches bidirectional models | Second-pass tokens get full context for pooling (Echo 2025) |
| Repeat > 3 times | Degradation | Attention sink disruption, distribution shift (Yona 2025, Xu 2024) |
| Pad with non-semantic tokens | No improvement | Semantic content of repetition is essential (Leviathan 2025) |

---

## 5. The Mechanism in Detail

### Why causal attention creates an information asymmetry

In a standard causal transformer with sequence `[t1, t2, t3, t4, t5]`:

```
t1 can see: [t1]
t2 can see: [t1, t2]
t3 can see: [t1, t2, t3]
t4 can see: [t1, t2, t3, t4]
t5 can see: [t1, t2, t3, t4, t5]
```

There is a severe information asymmetry: t1 has 1/5 the context of t5. The representation of t1 is committed with no knowledge of what follows. For tasks where the relationship between early and late tokens matters (which is most NLU tasks), this is a fundamental limitation.

### How repetition resolves the asymmetry

With repetition `[t1, t2, t3, t4, t5, t1', t2', t3', t4', t5']`:

```
t1' can see: [t1, t2, t3, t4, t5, t1']
t2' can see: [t1, t2, t3, t4, t5, t1', t2']
...
t5' can see: [t1, t2, t3, t4, t5, t1', t2', t3', t4', t5']
```

Now every token in the second pass has access to the complete first pass. The minimum context for any second-pass token is the full original sequence. This approximates bidirectional attention for the repeated portion.

### Why the second pass dominates

The model generates its response after the full repeated prompt. The last tokens processed before generation are the second-pass tokens, which have the richest representations (full bidirectional context). The model's final hidden state before generation thus reflects a more complete understanding of the query than a single-pass approach.

### The prefill efficiency argument

In modern LLM inference, the prefill stage (processing the input prompt) is highly parallelized -- all input tokens are processed simultaneously. The generation stage (producing output tokens) is sequential and latency-bound. Since repetition only extends the prefill, the wall-clock latency impact is minimal for most prompt lengths. The generation stage produces the same number of tokens as baseline.

---

## 6. Theoretical Connections

### Connection to bidirectional models (BERT, T5)

Bidirectional models like BERT use full attention masks where every token attends to every other token. This gives superior performance on understanding tasks but prevents autoregressive generation. Prompt repetition achieves an approximation of bidirectional attention within a causal framework -- the second pass "simulates" bidirectional attention by having the first pass available as context.

### Connection to Prefix LM

The T5 model uses a Prefix LM architecture where a designated prefix region uses bidirectional attention while the remainder uses causal attention. Prompt repetition effectively creates a similar structure: the first copy acts as a fully-observed prefix, and the second copy processes it with causal attention but full access to the prefix.

### Connection to encoder-decoder architectures

Encoder-decoder models (original Transformer, T5) process the input with a bidirectional encoder and generate output with a causal decoder cross-attending to the encoder. Prompt repetition in a decoder-only model approximates this: the first pass acts as a noisy "encoder" that the second pass "cross-attends" to via causal self-attention.

### Information-theoretic perspective

From an information-theoretic standpoint, a single causal pass compresses the input into a representation where early tokens are information-starved. Repetition provides redundancy that allows the model to recover information that was "lost" due to the causal bottleneck. This is analogous to error-correcting codes: the redundant copy allows recovery of information that couldn't be captured in a single pass.

---

## 7. Limitations and Open Questions

### Known limitations

1. **Length penalty tension:** Repetition doubles input length. For very long prompts, the length-induced degradation (Levy et al. 2024) may outweigh the bidirectional benefit.

2. **Diminishing returns with repetition count:** More than 2-3 repetitions degrades performance (Xu et al. 2024), likely due to attention sink disruption (Yona et al. 2025) and distribution shift from pretraining data.

3. **Reasoning tasks are neutral:** When step-by-step reasoning is enabled, models often naturally re-examine the query as part of their chain-of-thought, making explicit repetition redundant.

4. **Very long prompts may be impractical:** For prompts near the context window limit, repetition is impossible or requires truncation.

5. **The mechanism is model-dependent:** Different models may benefit to different degrees depending on their architecture, training data, and instruction tuning.

### Open questions

1. **Can models be fine-tuned to internalize repetition?** If trained on repeated prompts, would models learn to always "re-read" internally, eliminating the need for explicit repetition?

2. **Is partial repetition optimal for long prompts?** For a 10,000-token prompt, repeating only the last 500 tokens (the actual question + recent context) may be more effective than full repetition.

3. **How does repetition interact with system prompts?** Modern LLMs have system prompts, user prompts, and assistant turns. Which parts benefit most from repetition?

4. **What happens at the attention level during generation?** Do models primarily attend to the second copy during generation, effectively treating the first copy as disposable context?

5. **Can we predict when repetition will help?** Is there a measurable property of a prompt (e.g., information distribution, token-order sensitivity) that predicts whether repetition will improve performance?

6. **How does this interact with KV-cache optimizations?** If using KV-cache compression (e.g., Attention Matching, Selective Attention), does the compressed cache from the first copy provide the same bidirectional benefit?

---

## 8. Proposed Research Directions

The following directions extend prompt repetition into unstudied territory. Each builds on the established mechanism (causal attention asymmetry resolved via redundancy) and connects to a body of existing work that has NOT yet been linked to prompt repetition.

### 8.1 When to Repeat: The Length-Benefit Crossover

**Core question:** At what prompt length does the cost of repetition (length penalty) overtake the benefit (bidirectional context)?

**Background:** Levy et al. (2024) showed reasoning degrades with input length, even at 3000 tokens. Leviathan et al. (2025) showed repetition helps. These two forces are in direct tension, but nobody has mapped the crossover point.

**Proposed experiments:**
- Take a fixed task (e.g., multi-choice QA) and embed it in prompts of increasing length (500, 1000, 2000, 4000, 8000 tokens)
- At each length, compare baseline vs. repeated prompt
- Plot the "repetition delta" (accuracy_repeated - accuracy_baseline) against prompt length
- Hypothesis: delta is strongly positive for short prompts, crosses zero somewhere around 3000-5000 tokens, and becomes negative for very long prompts
- Test across multiple models to see if the crossover point is model-dependent

**Why it matters:** This would give practitioners a simple rule: "repeat if your prompt is under X tokens." No existing work provides this guidance.

**Connected work:**
- Levy, Jacoby, & Goldberg (2024). Same Task, More Tokens. arXiv:2402.14848
- Liu et al. (2023). Lost in the Middle. arXiv:2307.03172

### 8.2 What to Repeat: Partial Repetition Strategies

**Core question:** When full repetition is too expensive, which parts of the prompt should be repeated?

**Background:** Shaier et al. (2024) showed repeating only the question does nothing. Leviathan et al. (2025) showed repeating everything works. The space between these two extremes is completely unexplored.

**Modern LLM prompts have distinct regions:**
```
[SYSTEM PROMPT] [CONTEXT/DOCUMENTS] [CONVERSATION HISTORY] [CURRENT USER QUERY]
```

**Proposed experiments:**
- Test repeating each region independently and in combinations:
  - System prompt only (repeat safety/role instructions)
  - Context + query (skip system prompt)
  - Query + last assistant turn (conversational anchor)
  - Last N tokens only (sliding window repetition)
  - Extracted entities/keywords only (semantic compression)
- Compare against full repetition and baseline on a shared benchmark
- Measure the accuracy-per-added-token efficiency of each strategy

**Hypotheses:**
- Repeating the query alone fails (confirmed by Shaier) because the context-query attention gap isn't bridged
- Repeating context + query should capture most of the benefit since this bridges the primary attention gap
- Repeating the system prompt alone may help for instruction-following but not for factual QA
- A "bookend" strategy (repeat query before AND after context) may be optimal for long-context tasks

**Why it matters:** Full repetition doubles cost. Partial repetition could achieve 80% of the benefit at 20% of the cost. This is the practical question every engineer deploying this technique will ask.

### 8.3 Repetition as a Lost-in-the-Middle Fix

**Core question:** Can strategic repetition rescue information that LLMs lose from the middle of long contexts?

**Background:** Liu et al. (2023) demonstrated the "lost-in-the-middle" phenomenon: LLMs strongly prefer information at the beginning and end of their context, with a U-shaped performance curve. Information placed in the middle is effectively ignored. This has been confirmed across many models and tasks. CoRe (Park et al., 2024) showed that repeating context helps multi-hop QA, but didn't frame it as a lost-in-the-middle mitigation.

**Proposed experiments:**
- Use the "needle in a haystack" evaluation framework
- Place critical information at beginning, middle, and end positions
- Test three strategies:
  1. Baseline (no repetition)
  2. Full context repetition
  3. Targeted repetition (repeat only the middle third of the context)
- Measure whether repetition flattens the U-shaped performance curve
- Analyze attention patterns: does the second pass attend more evenly across positions?

**Hypotheses:**
- Full repetition flattens the U-curve because middle-positioned information in the first pass becomes early/accessible information in the second pass
- Targeted middle repetition may be more efficient than full repetition for long contexts
- The effect should be strongest for models with the most severe positional bias

**Why it matters:** Lost-in-the-middle is one of the most cited practical limitations of LLMs. If repetition can mitigate it without architectural changes or fine-tuning, that's immediately deployable.

**Connected work:**
- Liu et al. (2023). Lost in the Middle. arXiv:2307.03172
- Park et al. (2024). Context Repetition for Multi-Hop Reasoning. arXiv:2410.07103
- Levy et al. (2024). Found in the Middle: Plug-and-Play Positional Encoding. arXiv:2403.04797

### 8.4 Multi-Turn Dialogue: Repeating Across Conversation Turns

**Core question:** In multi-turn conversations, does repeating the user's current query at the end of accumulated conversation history improve response quality?

**Background:** Multi-turn dialogue accumulates context:
```
[System] You are a helpful assistant.
[User] What's my account balance?
[Assistant] Your balance is $1,234.
[User] What about my savings account?
[Assistant] Your savings balance is $5,678.
[User] Can you transfer $500 from savings to checking?
```

The current user query ("Can you transfer...") is at the end, but critical context ("savings account," "$5,678") is buried in the middle. This is exactly the lost-in-the-middle problem applied to conversation.

A recent paper "Do LLMs Benefit From Their Own Words?" (2026, arXiv:2602.24287) questions whether keeping assistant responses in history even helps, finding mixed results.

**Proposed experiments:**
- Simulate multi-turn conversations of increasing depth (3, 5, 10, 20 turns)
- Test strategies:
  1. Standard conversation history
  2. Repeat current user query at end
  3. Repeat current query + relevant prior context
  4. Summarize-then-repeat (compress history, repeat current query)
- Measure task completion accuracy, entity retention, and instruction following
- Test on task-oriented dialogue benchmarks (e.g., MultiWOZ-style tasks)

**Hypotheses:**
- Performance degrades with conversation depth (established)
- Repeating the current query helps because it reinforces the user's latest intent against accumulated noise
- The benefit increases with conversation depth (more middle-buried information to rescue)
- Repeating relevant prior context (not the whole history) is more effective than full repetition

**Why it matters:** Every chatbot and voice assistant faces this problem. A simple prompting strategy that improves multi-turn coherence without architectural changes would be widely adopted. This is also directly relevant to customer service bots where conversations can span 20+ turns.

**Connected work:**
- "Do LLMs Benefit From Their Own Words?" (2026). arXiv:2602.24287
- Park et al. (2024). Beyond Single-Turn: Multi-Turn Interactions Survey. arXiv:2504.04717
- Liu et al. (2023). Lost in the Middle. arXiv:2307.03172

### 8.5 Robustness to Noisy Inputs (ASR and Beyond)

**Core question:** Does prompt repetition improve LLM performance when inputs contain noise -- from ASR errors, OCR artifacts, typos, or non-native speaker patterns?

**Background:** Voice-based LLM applications pipe ASR transcriptions into the model. These transcriptions contain errors: substitutions ("recognize" -> "wreck a nice"), deletions, insertions, and disfluencies ("um," "uh," false starts). A body of work exists on making NLU robust to ASR noise (ASR-GLUE, Noise-BERT, Confusion2Vec, MEDSAGE), but none has tested whether simple prompt repetition helps.

The mechanism is plausible: if the model sees a noisy token in the first pass, the surrounding context from the full first pass may help the second-pass representation "correct" or disambiguate the error. This is analogous to how humans re-read garbled text and use surrounding words to infer meaning.

**Proposed experiments:**
- Take standard QA benchmarks and inject realistic noise:
  - ASR-style phonetic substitutions (using phonetic confusion matrices)
  - Random character-level typos (1%, 5%, 10% error rates)
  - Disfluency injection (filled pauses, restarts, corrections)
  - Non-native speaker patterns (dropped articles, wrong prepositions)
- Compare baseline vs. repeated prompt at each noise level
- Plot accuracy vs. noise level for both conditions
- Test whether repetition provides more benefit at higher noise levels

**Hypotheses:**
- Repetition helps more as noise increases, because the second-pass attention can use clean surrounding tokens from the first pass to disambiguate noisy tokens
- The benefit is strongest for substitution errors (where context can disambiguate) and weakest for deletion errors (where information is genuinely missing)
- At very high noise levels, repetition may hurt because it doubles the noise exposure

**Why it matters:** Voice interfaces are growing rapidly. If prompt repetition is a zero-cost way to improve robustness to ASR errors, it could be deployed in every voice assistant and call center bot immediately.

**Connected work:**
- Huang et al. (2024). ASR-GLUE: Multi-task Benchmark for ASR-Robust NLU. arXiv:2108.13048
- Park et al. (2024). MEDSAGE: Robustness of Medical Dialogue Summarization to ASR Errors. arXiv:2408.14418
- Shin et al. (2024). Speak & Spell: Phonetic Error Augmentation for Robust DST. arXiv:2409.06263
- Chen et al. (2020). Warped Language Models for Noise Robust Language Understanding. arXiv:2011.01900

### 8.6 Repetition as Prompt Injection Defense

**Core question:** Can repeating safety-critical instructions after user input make LLMs more resistant to prompt injection attacks?

**Background:** Prompt injection exploits the causal attention asymmetry: system instructions appear early in the context, user input appears later. Since later tokens have richer representations (they attend to everything), adversarial user content can override system instructions. The system prompt is "information-starved" relative to the user's injection.

Repeating the system prompt AFTER the user input would flip this dynamic: the repeated system instructions become the final context before generation, with full attention to everything including the attack.

**Proposed experiments:**
- Establish a baseline prompt injection success rate using known attack datasets
- Test three defense strategies:
  1. Standard: `[System] [User Input]`
  2. System repeat: `[System] [User Input] [System]`
  3. Selective repeat: `[System] [User Input] [Key safety instructions only]`
- Measure both defense success rate AND task performance (to check for over-refusal)
- Compare against known defenses (input filtering, output filtering, fine-tuned safety)

**Hypotheses:**
- Repeating system instructions after user input significantly reduces injection success because the model's final representations are dominated by the safety instructions
- This is more effective than just making system prompts longer or more emphatic
- There's a tradeoff: too much repetition of safety instructions may increase refusal rates on benign queries

**Why it matters:** Prompt injection is one of the top security concerns for deployed LLMs. A zero-cost, zero-latency defense that requires no model modification would be extremely valuable. No existing work has connected repetition to adversarial robustness.

**Connected work:**
- Yi et al. (2023). Evaluating Instruction-Following Robustness to Prompt Injection. arXiv:2308.10819
- Willison et al. (2024). Baseline Defenses for Adversarial Attacks. arXiv:2309.00614
- Leviathan et al. (2025). Selective Attention (same authors -- architectural approach to the same problem). arXiv:2410.02703

### 8.7 The Redundancy-Selection Spectrum

**Core question:** Can we unify prompt repetition (brute-force redundancy) and selective attention (intelligent forgetting) into a single theoretical framework?

**Background:** Leviathan et al. authored both Prompt Repetition and Selective Attention. These are opposite approaches to the same problem:
- **Prompt Repetition:** Add redundant information so the model can attend to everything
- **Selective Attention:** Remove irrelevant information so the model focuses on what matters

Both improve performance. Both address the causal attention bottleneck. But they operate at different ends of a spectrum, and no work has studied them together.

**Proposed framework:** Information-theoretic analysis of the tradeoff:
- Define "effective attention coverage" as the fraction of input information accessible to the model during generation
- Single pass: coverage is asymmetric (late tokens have high coverage, early tokens have low coverage)
- Repetition: coverage approaches 100% but at 2x token cost
- Selective attention: coverage focuses on relevant information, reducing noise
- Optimal strategy: repeat only the information that selective attention would preserve

**Proposed experiments:**
- Train small models with selective attention and analyze which tokens get masked
- Test whether repeating only the tokens that survive selective attention (i.e., the "important" tokens) outperforms full repetition
- Compare four conditions: baseline, full repetition, selective attention, selective repetition

**Why it matters:** This provides the theoretical foundation connecting two empirically successful techniques. It also suggests a practical algorithm: use selective attention to identify what matters, then repeat only that.

**Connected work:**
- Leviathan et al. (2025). Selective Attention. arXiv:2410.02703
- Leviathan et al. (2025). Prompt Repetition. arXiv:2512.14982
- Information-theoretic perspectives on attention: Poli et al. (2023), Massaroli et al. (2020)

### 8.8 Paraphrased vs. Verbatim Repetition

**Core question:** Is verbatim repetition optimal, or would paraphrasing the repeated copy provide additional benefit through lexical diversity?

**Background:** All existing repetition work uses verbatim copies. But human learning benefits from seeing the same concept expressed in multiple ways. Paraphrasing could provide:
- Different tokenization of the same concepts (exposing different token-level representations)
- Emphasis on different aspects of the query
- Reduced risk of attention sink issues from identical token sequences (Yona et al. 2025)

However, Shaier et al. (2024) found that paraphrased question repetition sometimes HURT performance for larger models. This may be because paraphrasing introduces ambiguity or because the model treats the paraphrase as a different question.

**Proposed experiments:**
- Generate paraphrases of test prompts using a separate LLM
- Compare: baseline vs. verbatim repetition vs. paraphrased repetition vs. verbatim + paraphrased
- Control for paraphrase quality (human-rated meaning preservation)
- Analyze whether the paraphrase effect depends on task type (factual QA vs. subjective tasks)

**Hypotheses:**
- Verbatim repetition wins for factual/extraction tasks (exact token matching matters)
- Paraphrased repetition may win for tasks requiring deeper understanding (the diverse framing helps)
- Combined (verbatim + paraphrase) may provide the best of both worlds but at 3x cost

**Why it matters:** If paraphrasing helps, it suggests the benefit is not just mechanical (causal attention fix) but also cognitive (richer representation from diverse framing). This would have implications for how we think about prompt engineering more broadly.

---

## 9. A Unified Narrative

These eight directions share a common thread: **the causal attention mask creates an information asymmetry that can be partially resolved through strategic redundancy.** The key insight is that this isn't a single technique but a design space with multiple axes:

| Axis | Range |
|------|-------|
| **What to repeat** | Nothing ... question only ... context + question ... everything |
| **How many times** | 1 (baseline) ... 2 (sweet spot) ... 3+ (diminishing/negative returns) |
| **Verbatim vs. paraphrased** | Exact copy ... light rephrase ... full paraphrase |
| **When to repeat** | Always ... only for long prompts ... only for noisy inputs ... only for safety-critical |
| **Where in the prompt** | Prepend ... append ... bookend ... interleave |
| **Cost model** | Free (short prompts) ... moderate (medium prompts) ... prohibitive (near context limit) |

The unexplored regions of this design space represent genuine research opportunities. The most impactful work would map this space empirically and provide practitioners with clear guidelines: given your prompt length, task type, noise level, and latency budget, here is the optimal repetition strategy.

---

## 10. References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2025). Prompt Repetition Improves Non-Reasoning LLMs. arXiv:2512.14982.
2. Xu, X., Tao, C., Shen, T., Xu, C., Xu, H., Long, G., Lou, J. (2024). Re-Reading Improves Reasoning in Large Language Models. arXiv:2309.06275.
3. Springer, J.M., Kotha, S., Fried, D., Neubig, G., & Raghunathan, A. (2025). Repetition Improves Language Model Embeddings. ICLR 2025. arXiv:2402.15449.
4. Shaier, S., Sanz-Guerrero, M., & von der Wense, K. (2024). Asking Again and Again: Exploring LLM Robustness to Repeated Questions. arXiv:2412.07923.
5. Park, J. et al. (2024). Unleashing Multi-Hop Reasoning Potential in Large Language Models through Repetition of Misordered Context. arXiv:2410.07103.
6. Levy, M., Jacoby, A., & Goldberg, Y. (2024). Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models. arXiv:2402.14848.
7. Leviathan, Y., Kalman, M., & Matias, Y. (2025). Selective Attention Improves Transformer. ICLR 2025. arXiv:2410.02703.
8. Yona, I., Shumailov, I., Hayes, J., Barbero, F., & Gandelsman, Y. (2025). Interpreting the Repeated Token Phenomenon in Large Language Models. arXiv:2503.08908.
9. Raffel, C. et al. (2023). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. arXiv:1910.10683.
10. Liu, N.F. et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. arXiv:2307.03172.
11. "Do LLMs Benefit From Their Own Words?" (2026). arXiv:2602.24287.
12. Huang et al. (2021). ASR-GLUE: Multi-task Benchmark for ASR-Robust NLU. arXiv:2108.13048.
13. Park et al. (2024). MEDSAGE: Robustness of Medical Dialogue Summarization to ASR Errors. arXiv:2408.14418.
14. Shin et al. (2024). Speak & Spell: Phonetic Error Augmentation for Robust DST. arXiv:2409.06263.
15. Chen et al. (2020). Warped Language Models for Noise Robust Language Understanding. arXiv:2011.01900.
16. Yi et al. (2023). Evaluating Instruction-Following Robustness to Prompt Injection. arXiv:2308.10819.
17. Levy et al. (2024). Found in the Middle: Plug-and-Play Positional Encoding. arXiv:2403.04797.
18. Sequence Repetition Enhances Token Embeddings for Sequence Labeling (2026). arXiv:2601.07894.

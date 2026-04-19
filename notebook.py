# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import json
    import sys
    from pathlib import Path
    return np, plt, mcolors, json, sys, Path


# ============================================================
# Title & Introduction
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Prompt Repetition: When and What to Repeat

    An interactive exploration of [**Prompt Repetition Improves Non-Reasoning LLMs**](https://arxiv.org/abs/2512.14982)
    (Leviathan, Kalman & Matias, Google Research, 2025).

    **The core idea:** Decoder-only LLMs use causal attention -- each token can only attend to
    tokens that came *before* it. By repeating the prompt, every token in the second copy gets
    full bidirectional context from the first copy, approximating what encoder models get for free.

    This notebook demonstrates the mechanism with **interactive visualizations** and presents
    results from experiments across multiple models and benchmarks.

    ---

    **Contents:**
    1. Why it works: the causal attention bottleneck (interactive)
    2. Token information coverage (interactive)
    3. Experiment results across model sizes
    4. Partial repetition strategies
    5. Noisy inputs: where repetition fails
    6. Related work
    """)
    return


# ============================================================
# Load pre-computed results
# ============================================================

@app.cell
def _(json, mo, sys, Path):
    def _load_results():
        if "pyodide" in sys.modules:
            _p = Path(mo.notebook_location()) / "public" / "results.json"
        else:
            _p = Path("public/results.json")
            if not _p.exists():
                _p = Path("data/results.json")
        if _p.exists():
            with open(_p) as _f:
                return json.load(_f)
        return {}

    RESULTS = _load_results()
    return (RESULTS,)


# ============================================================
# DEMO A: Attention Mask Visualizer
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. The Causal Attention Bottleneck

    In a decoder-only transformer, token $t_i$ can only attend to $\{t_1, \ldots, t_i\}$ (causal mask).
    This means early tokens are **information-starved** -- $t_1$ sees only itself, $t_2$ sees two tokens, etc.

    When you **repeat the prompt**, every token in the second copy sees the *entire* first copy.
    The second copy of $t_1$ now has the same context as $t_n$ did in the original -- full bidirectional view.

    **Try it below:** type a sentence and toggle repetition to see how the attention mask changes.
    """)
    return


@app.cell
def _(mo):
    attn_input = mo.ui.text(
        value="What color is the sky on Mars",
        label="Enter a prompt:",
        full_width=True,
    )
    attn_repeat = mo.ui.checkbox(value=False, label="Repeat prompt")
    mo.hstack([attn_input, attn_repeat], gap=1)
    return attn_input, attn_repeat


@app.cell
def _(attn_input, attn_repeat, np, plt, mcolors, mo):
    _words = attn_input.value.split()
    if attn_repeat.value:
        _tokens = _words + ["|"] + _words
    else:
        _tokens = _words

    _n = len(_tokens)
    _mask = np.zeros((_n, _n))
    _sep = len(_words)

    for _i in range(_n):
        for _j in range(_i + 1):
            _mask[_i, _j] = 1.0

    _cmap = mcolors.ListedColormap(["#f0f0f0", "#4a90d9"])

    _fig, _axes = plt.subplots(1, 2 if not attn_repeat.value else 1,
                                figsize=(12 if not attn_repeat.value else 8, 5))

    if not attn_repeat.value:
        # Show causal mask + what bidirectional would look like
        _ax1 = _axes[0]
        _ax1.imshow(_mask, cmap=_cmap, aspect="equal", interpolation="nearest")
        _ax1.set_xticks(range(_n))
        _ax1.set_yticks(range(_n))
        _ax1.set_xticklabels(_tokens, rotation=45, ha="right", fontsize=8)
        _ax1.set_yticklabels(_tokens, fontsize=8)
        _ax1.set_title("Causal Attention (what LLMs actually use)", fontsize=11, fontweight="bold")
        _ax1.set_xlabel("Keys (can attend to)")
        _ax1.set_ylabel("Queries (each token)")

        _bi_mask = np.ones((_n, _n))
        _ax2 = _axes[1]
        _ax2.imshow(_bi_mask, cmap=_cmap, aspect="equal", interpolation="nearest")
        _ax2.set_xticks(range(_n))
        _ax2.set_yticks(range(_n))
        _ax2.set_xticklabels(_tokens, rotation=45, ha="right", fontsize=8)
        _ax2.set_yticklabels(_tokens, fontsize=8)
        _ax2.set_title("Bidirectional (what encoders use)", fontsize=11, fontweight="bold")
        _ax2.set_xlabel("Keys (can attend to)")
        _ax2.set_ylabel("Queries (each token)")
    else:
        _ax = _axes if not hasattr(_axes, '__len__') else _axes
        # Color the second copy's access to the first copy
        _colors = np.zeros((_n, _n, 3))
        for _i in range(_n):
            for _j in range(_i + 1):
                if _i >= _sep + 1 and _j < _sep:
                    _colors[_i, _j] = [0.2, 0.7, 0.3]  # green = NEW access
                elif _mask[_i, _j] > 0:
                    _colors[_i, _j] = [0.29, 0.56, 0.85]  # blue = original access
                else:
                    _colors[_i, _j] = [0.94, 0.94, 0.94]

        _ax.imshow(_colors, aspect="equal", interpolation="nearest")
        _ax.set_xticks(range(_n))
        _ax.set_yticks(range(_n))
        _ax.set_xticklabels(_tokens, rotation=45, ha="right", fontsize=7)
        _ax.set_yticklabels(_tokens, fontsize=7)
        _ax.set_title("Repeated Prompt: Causal Attention", fontsize=11, fontweight="bold")
        _ax.set_xlabel("Keys (can attend to)")
        _ax.set_ylabel("Queries (each token)")
        _ax.axhline(y=_sep + 0.5, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
        _ax.axvline(x=_sep - 0.5, color="red", linewidth=1.5, linestyle="--", alpha=0.7)

        from matplotlib.patches import Patch
        _ax.legend(
            handles=[
                Patch(color=[0.29, 0.56, 0.85], label="Original causal access"),
                Patch(color=[0.2, 0.7, 0.3], label="NEW: 2nd copy sees 1st copy"),
            ],
            loc="upper right", fontsize=8,
        )

    plt.tight_layout()
    mo.md(f"""
    **Context coverage per token position:**
    - Without repetition: token 1 sees {1}/{len(_words)} words ({1/len(_words)*100:.0f}%), token {len(_words)} sees {len(_words)}/{len(_words)} (100%)
    - With repetition: every token in the 2nd copy sees all {len(_words)} original words (100%)
    """)
    _fig
    return


# ============================================================
# DEMO B: Token Information Coverage
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2. Token Information Coverage

    How much context does each token position "see"? In a causal model, early tokens are starved.
    Repetition flattens the curve -- every position in the second copy has near-complete coverage.

    Adjust the prompt length to see how the coverage gap scales.
    """)
    return


@app.cell
def _(mo):
    coverage_n = mo.ui.slider(5, 50, 1, value=15, label="Prompt length (tokens):")
    coverage_n
    return (coverage_n,)


@app.cell
def _(coverage_n, np, plt, mo):
    _n = coverage_n.value

    # Causal: token i sees (i+1) out of n tokens
    _causal = np.array([(i + 1) / _n for i in range(_n)])

    # Repeated: first copy same as causal, second copy sees all of first + causal within second
    _rep_first = _causal.copy()
    _rep_second = np.array([(_n + i + 1) / (2 * _n) for i in range(_n)])
    _repeated = np.concatenate([_rep_first, _rep_second])

    # Padded: same length as repeated, but padding adds no information
    _pad_coverage = np.array([(i + 1) / _n if i < _n else 1.0 for i in range(2 * _n)])

    # "Effective" coverage: what fraction of MEANINGFUL tokens each position sees
    _causal_eff = _causal
    _rep_eff = np.concatenate([_causal, np.ones(_n)])  # 2nd copy sees all original tokens
    _pad_eff = np.concatenate([_causal, _causal])  # padding doesn't help

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: raw coverage
    _ax1.plot(range(_n), _causal, "o-", color="#e74c3c", label="Single prompt", markersize=4, linewidth=2)
    _ax1.plot(range(2 * _n), _repeated, "s-", color="#2ecc71", label="Repeated prompt", markersize=3, linewidth=2)
    _ax1.plot(range(2 * _n), _pad_coverage, "^-", color="#95a5a6", label="Padded (length control)", markersize=3, linewidth=1.5, alpha=0.7)
    _ax1.axhline(y=1.0, color="black", linewidth=0.5, linestyle=":", alpha=0.3)
    _ax1.axvline(x=_n - 0.5, color="red", linewidth=1, linestyle="--", alpha=0.4, label="Prompt boundary")
    _ax1.set_xlabel("Token position")
    _ax1.set_ylabel("Fraction of total sequence visible")
    _ax1.set_title("Raw Attention Coverage", fontweight="bold")
    _ax1.legend(fontsize=8)
    _ax1.set_ylim(-0.05, 1.1)

    # Right: effective coverage (what fraction of UNIQUE information is visible)
    _ax2.plot(range(_n), _causal_eff, "o-", color="#e74c3c", label="Single prompt", markersize=4, linewidth=2)
    _ax2.plot(range(2 * _n), _rep_eff, "s-", color="#2ecc71", label="Repeated prompt", markersize=3, linewidth=2)
    _ax2.plot(range(2 * _n), _pad_eff, "^-", color="#95a5a6", label="Padded", markersize=3, linewidth=1.5, alpha=0.7)
    _ax2.axhline(y=1.0, color="black", linewidth=0.5, linestyle=":", alpha=0.3)
    _ax2.axvline(x=_n - 0.5, color="red", linewidth=1, linestyle="--", alpha=0.4)
    _ax2.set_xlabel("Token position")
    _ax2.set_ylabel("Fraction of unique prompt info visible")
    _ax2.set_title("Effective Information Coverage", fontweight="bold")
    _ax2.legend(fontsize=8)
    _ax2.set_ylim(-0.05, 1.1)
    _ax2.annotate("2nd copy: full coverage!",
                  xy=(int(1.5 * _n), 1.0), fontsize=9, color="#2ecc71", fontweight="bold",
                  ha="center", va="bottom")

    plt.tight_layout()

    _avg_causal = np.mean(_causal_eff)
    _avg_rep = np.mean(_rep_eff)
    mo.md(f"""
    **Average effective coverage:** Single = {_avg_causal:.1%} | Repeated = {_avg_rep:.1%} | Gain = **{_avg_rep - _avg_causal:+.1%}**

    The right panel shows the key insight: padding adds length but no new information,
    so the effective coverage stays the same. Repetition gives every token in the second copy
    access to the full original prompt -- a {_avg_rep - _avg_causal:+.1%} improvement in average coverage.
    """)
    _fig
    return


# ============================================================
# DEMO C: Experiment 1 Results -- Model Scaling Curve
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3. Experiment Results: Does Model Size Matter?

    We tested prompt repetition on **ARC Challenge** (4-choice science, n=100) and **MMLU-Pro**
    (10-choice multi-domain, n=100) across 4 models spanning 1B to frontier-tier parameters.

    For each question, we compare three conditions:
    - **Baseline:** single prompt
    - **Repeated:** prompt concatenated with itself
    - **Padded:** prompt + equivalent-length meaningless filler (controls for input length)
    """)
    return


@app.cell
def _(mo):
    bench_select = mo.ui.dropdown(
        options={"ARC Challenge": "ARC", "MMLU-Pro": "MMLU", "Both (averaged)": "both"},
        value="ARC Challenge",
        label="Benchmark:"
    )
    bench_select
    return (bench_select,)


@app.cell
def _(bench_select, RESULTS, np, plt, mo):
    _exp1 = RESULTS.get("experiment_1", {})

    _models = [
        ("Llama 3.2 1B", 1),
        ("Llama 3.1 8B", 8),
        ("Llama 4 Scout 17B", 17),
        ("GPT-4o Mini", 30),  # approximate
    ]

    _bench = bench_select.value
    _sizes = []
    _baselines = []
    _repeated = []
    _padded = []
    _deltas = []
    _names = []

    for _name, _size in _models:
        _data = _exp1.get(_name, {})
        if _bench == "both":
            _arc = _data.get("ARC", {})
            _mmlu = _data.get("MMLU", {})
            if _arc and _mmlu:
                _b = (_arc.get("baseline", 0) + _mmlu.get("baseline", 0)) / 2
                _r = (_arc.get("repeated", 0) + _mmlu.get("repeated", 0)) / 2
                _p = (_arc.get("padded", 0) + _mmlu.get("padded", 0)) / 2
            else:
                continue
        else:
            _d = _data.get(_bench, {})
            if not _d:
                continue
            _b = _d.get("baseline", 0)
            _r = _d.get("repeated", 0)
            _p = _d.get("padded", 0)

        _sizes.append(_size)
        _baselines.append(_b)
        _repeated.append(_r)
        _padded.append(_p)
        _deltas.append(_r - _b)
        _names.append(_name)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy by condition
    _x = np.arange(len(_names))
    _w = 0.25
    _ax1.bar(_x - _w, _baselines, _w, label="Baseline", color="#e74c3c", alpha=0.85)
    _ax1.bar(_x, _repeated, _w, label="Repeated", color="#2ecc71", alpha=0.85)
    _ax1.bar(_x + _w, _padded, _w, label="Padded", color="#95a5a6", alpha=0.85)
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(_names, rotation=15, ha="right", fontsize=9)
    _ax1.set_ylabel("Accuracy (%)")
    _ax1.set_title(f"Accuracy by Condition ({_bench if _bench != 'both' else 'Averaged'})", fontweight="bold")
    _ax1.legend(fontsize=9)
    _ax1.set_ylim(0, 100)

    for _i, (_b, _r) in enumerate(zip(_baselines, _repeated)):
        _delta = _r - _b
        _color = "#2ecc71" if _delta > 0 else "#e74c3c"
        _ax1.annotate(f"{_delta:+.0f}%", xy=(_i, max(_b, _r) + 2),
                      ha="center", fontsize=9, fontweight="bold", color=_color)

    # Right: delta vs baseline accuracy (sweet spot analysis)
    _ax2.scatter(_baselines, _deltas, s=120, c=[_d if _d >= 0 else _d for _d in _deltas],
                 cmap="RdYlGn", edgecolors="black", linewidth=0.5, zorder=5,
                 vmin=-10, vmax=50)
    for _i, _name in enumerate(_names):
        _ax2.annotate(_name, (_baselines[_i], _deltas[_i]),
                      textcoords="offset points", xytext=(8, 5), fontsize=8)

    _ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="-", alpha=0.3)
    _ax2.axvspan(25, 70, alpha=0.08, color="green", label="Sweet spot (30-70% baseline)")
    _ax2.set_xlabel("Baseline Accuracy (%)")
    _ax2.set_ylabel("Repetition Delta (pp)")
    _ax2.set_title("Where Repetition Helps Most", fontweight="bold")
    _ax2.legend(fontsize=8)

    plt.tight_layout()

    _best = _names[np.argmax(_deltas)] if _deltas else "N/A"
    _best_delta = max(_deltas) if _deltas else 0
    mo.md(f"""
    **Biggest winner: {_best}** with **{_best_delta:+.0f} percentage points** improvement.

    The right panel reveals the **sweet spot**: repetition helps most when baseline accuracy
    is 30-70%. Models that are too weak (near random) lack the capacity to benefit.
    Models that are too strong (85%+) hit ceiling effects. The padded control confirms
    this isn't just a length artifact -- repetition provides genuine information gain.
    """)
    _fig
    return


# ============================================================
# DEMO D: Partial Repetition Strategies
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4. Partial Repetition: What Should You Repeat?

    Not all repetition is equal. Shaier et al. (2024) found repeating *just the question*
    doesn't help. Leviathan et al. (2025) showed repeating *everything* does. But what about
    the space between? We tested 4 strategies on ARC questions embedded in context paragraphs.
    """)
    return


@app.cell
def _(RESULTS, plt, mo):
    _exp2 = RESULTS.get("experiment_2", {})
    _results = _exp2.get("results", {})
    _model = _exp2.get("model", "Llama 3.1 8B")

    _strategies = ["Baseline (no repeat)", "Repeat everything", "Repeat question only", "Question-Context-Question"]
    _short_labels = ["Baseline", "Repeat All", "Q Only", "Bookend Q"]
    _accs = [_results.get(s, {}).get("accuracy", 0) for s in _strategies]
    _base = _accs[0]
    _deltas = [a - _base for a in _accs]

    _colors = ["#95a5a6" if d == 0 else "#2ecc71" if d > 0 else "#e74c3c" for d in _deltas]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    _ax1.barh(range(len(_short_labels)), _accs, color=_colors, edgecolor="black", linewidth=0.5)
    _ax1.set_yticks(range(len(_short_labels)))
    _ax1.set_yticklabels(_short_labels, fontsize=10)
    _ax1.set_xlabel("Accuracy (%)")
    _ax1.set_title(f"Partial Repetition ({_model})", fontweight="bold")
    _ax1.axvline(x=_base, color="black", linewidth=1, linestyle="--", alpha=0.4)
    for _i, (_a, _d) in enumerate(zip(_accs, _deltas)):
        _ax1.text(_a + 0.5, _i, f"{_a:.0f}% ({_d:+.0f})", va="center", fontsize=9)

    # Right: diagram of each strategy
    _ax2.set_xlim(0, 10)
    _ax2.set_ylim(-0.5, 4.5)
    _ax2.axis("off")
    _ax2.set_title("Strategy Layouts", fontweight="bold", fontsize=11)

    _layouts = [
        ("Baseline", ["Context", "Question"]),
        ("Repeat All", ["Context", "Question", "Context", "Question"]),
        ("Q Only", ["Context", "Question", "Question"]),
        ("Bookend Q", ["Question", "Context", "Question"]),
    ]
    _block_colors = {"Context": "#dbeafe", "Question": "#fef3c7"}

    for _row, (_label, _blocks) in enumerate(_layouts):
        _y = 3.5 - _row
        _x = 0.5
        _ax2.text(0.1, _y, _label, fontsize=8, va="center", fontweight="bold")
        _bw = (9.0 - _x) / max(len(_blocks), 1)
        for _b, _block in enumerate(_blocks):
            _rect = plt.Rectangle((_x + _b * _bw, _y - 0.25), _bw - 0.05, 0.5,
                                   facecolor=_block_colors.get(_block, "#f0f0f0"),
                                   edgecolor="black", linewidth=0.5)
            _ax2.add_patch(_rect)
            _ax2.text(_x + _b * _bw + _bw / 2, _y, _block, fontsize=7,
                      ha="center", va="center")

    plt.tight_layout()

    mo.md(f"""
    **Surprising finding:** Repeating *only the question* gave **+4%**, while repeating
    *everything* gave **0%**. This contradicts Shaier et al. (2024), who found question-only
    repetition ineffective, and challenges the assumption that more repetition = better.

    **Hypothesis:** With short context (3 paragraphs), repeating everything doubles the
    noise-to-signal ratio. Repeating just the question reinforces the decision-critical
    information without amplifying irrelevant context. This suggests an optimal repetition
    strategy depends on the context-to-question ratio.
    """)
    _fig
    return


# ============================================================
# DEMO E: Noisy Inputs
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 5. Where Repetition Fails: Noisy Inputs

    A natural question: if repetition helps with clean prompts, does it help with *noisy* prompts
    (like ASR transcriptions from voice bots)? We injected word-level errors (substitutions,
    deletions, disfluencies) at 0-20% rates and tested whether repetition recovers accuracy.
    """)
    return


@app.cell
def _(RESULTS, np, plt, mo):
    _exp4 = RESULTS.get("experiment_4", {})
    _results = _exp4.get("results", {})
    _model = _exp4.get("model", "Llama 3.1 8B")

    _rates = ["0%", "5%", "10%", "15%", "20%"]
    _rate_nums = [0, 5, 10, 15, 20]
    _clean = [_results.get(r, {}).get("clean", 0) for r in _rates]
    _noisy = [_results.get(r, {}).get("noisy", 0) for r in _rates]
    _noisy_rep = [_results.get(r, {}).get("noisy_rep", 0) for r in _rates]
    _deltas = [nr - n for nr, n in zip(_noisy_rep, _noisy)]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    _ax1.plot(_rate_nums, _clean, "o-", color="#3498db", label="Clean baseline", linewidth=2, markersize=6)
    _ax1.plot(_rate_nums, _noisy, "s-", color="#e74c3c", label="Noisy", linewidth=2, markersize=6)
    _ax1.plot(_rate_nums, _noisy_rep, "^-", color="#2ecc71", label="Noisy + Repeated", linewidth=2, markersize=6)
    _ax1.fill_between(_rate_nums, _noisy, _noisy_rep, alpha=0.15,
                       color="red" if np.mean(_deltas) < 0 else "green")
    _ax1.set_xlabel("Word Error Rate (%)")
    _ax1.set_ylabel("Accuracy (%)")
    _ax1.set_title(f"Noise Robustness ({_model})", fontweight="bold")
    _ax1.legend(fontsize=9)
    _ax1.set_ylim(50, 85)

    _colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in _deltas]
    _ax2.bar(_rate_nums, _deltas, width=3, color=_colors, edgecolor="black", linewidth=0.5)
    _ax2.axhline(y=0, color="black", linewidth=1)
    _ax2.set_xlabel("Word Error Rate (%)")
    _ax2.set_ylabel("Repetition Delta (pp)")
    _ax2.set_title("Effect of Repetition on Noisy Input", fontweight="bold")
    for _i, (_r, _d) in enumerate(zip(_rate_nums, _deltas)):
        _ax2.text(_r, _d - 0.5 if _d < 0 else _d + 0.2, f"{_d:+.0f}%",
                  ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()

    mo.md(f"""
    **Key negative result:** Repetition consistently **hurts** with noisy inputs ({np.mean(_deltas):+.1f}% average).

    This makes theoretical sense: the causal attention mechanism helps early tokens "see" later tokens,
    but if both copies contain the **same errors**, there's no clean signal to recover from.
    Repetition doubles the noise exposure rather than providing error correction.

    **Practical implication:** Prompt repetition should be avoided in voice bot / ASR pipelines
    where input noise is expected. It's a technique specifically for *clean* input scenarios.
    """)
    _fig
    return


# ============================================================
# Interactive: Noise Simulation
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Try it yourself: Noise Injection Demo

    See what ASR-style noise does to a prompt. The model sees this garbled text --
    repeating it just doubles the confusion.
    """)
    return


@app.cell
def _(mo):
    noise_input = mo.ui.text(
        value="What is the chemical formula for water",
        label="Clean prompt:",
        full_width=True,
    )
    noise_rate = mo.ui.slider(0, 50, 1, value=15, label="Error rate (%):")
    noise_seed = mo.ui.button(label="Re-roll noise", kind="neutral")
    mo.vstack([noise_input, mo.hstack([noise_rate, noise_seed])])
    return noise_input, noise_rate, noise_seed


@app.cell
def _(noise_input, noise_rate, noise_seed, np, mo):
    noise_seed

    _rng = np.random.default_rng()
    _words = noise_input.value.split()
    _rate = noise_rate.value / 100
    _noisy = []
    _changes = []

    for _w in _words:
        if _rng.random() < _rate and len(_w) > 3:
            _t = _rng.choice(["sub", "del", "dis"])
            if _t == "sub":
                _c = list(_w)
                _pos = _rng.integers(0, len(_c))
                _c[_pos] = chr(_rng.integers(97, 123))
                _nw = "".join(_c)
                _noisy.append(_nw)
                _changes.append(f"'{_w}' -> '{_nw}' (substitution)")
            elif _t == "del":
                _c = list(_w)
                _c.pop(_rng.integers(0, len(_c)))
                _nw = "".join(_c)
                _noisy.append(_nw)
                _changes.append(f"'{_w}' -> '{_nw}' (deletion)")
            else:
                _noisy.append("uh")
                _noisy.append(_w)
                _changes.append(f"'{_w}' -> 'uh {_w}' (disfluency)")
        else:
            _noisy.append(_w)

    _noisy_text = " ".join(_noisy)
    _n_errors = len(_changes)

    mo.md(f"""
    **Noisy version:** {_noisy_text}

    **Repeated (what the model sees):**
    > {_noisy_text}
    >
    > {_noisy_text}

    **Errors injected ({_n_errors}):** {'; '.join(_changes) if _changes else 'None'}

    Notice how repetition just shows the model the same errors twice -- no clean reference to correct against.
    """)
    return


# ============================================================
# Summary & Related Work
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 6. Summary & Related Work

    ### Key Takeaways

    1. **Prompt repetition works** -- but only in the sweet spot (30-70% baseline accuracy)
    2. **It's not just a length effect** -- the padded control confirms genuine information gain
    3. **What you repeat matters** -- question-only repetition can outperform full repetition
    4. **It fails with noisy inputs** -- repetition doubles noise exposure, hurting accuracy
    5. **The mechanism is causal attention** -- repetition approximates bidirectional context

    ### How This Connects to the Literature

    | Paper | Finding | Connection |
    |-------|---------|------------|
    | **Leviathan et al. 2025** | Full prompt repetition improves non-reasoning tasks | Core paper we reproduce |
    | Xu et al. 2024 (RE2) | Re-reading with "Read again:" instruction improves reasoning | Complementary: explicit instruction + repetition |
    | Springer et al. 2025 (Echo) | Repetition improves LM embeddings | Same mechanism, embedding task |
    | **Shaier et al. 2024** | Repeating just the question does NOT help | We partially contradict this (Exp 2) |
    | Park et al. 2024 (CoRe) | Context repetition fixes misordered multi-hop QA | Same mechanism, document ordering |
    | Levy et al. 2024 | Longer inputs can degrade reasoning | The cost side of the repetition tradeoff |
    | Liu et al. 2023 | Lost-in-the-middle positional bias | The problem repetition could address |
    | Yona et al. 2025 | Excessive repetition disrupts attention sinks | Upper bound on repetition count |

    ### Our Novel Contributions

    - **Model-size scaling curve**: First systematic mapping of where repetition helps vs. hurts across model sizes
    - **Partial repetition taxonomy**: Testing 4 strategies reveals question-only can beat full repetition
    - **Noise degradation curve**: First test of repetition with ASR-style noise -- important negative result
    - **Interactive attention visualization**: Makes the causal attention bottleneck intuitive and explorable
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    *Built with [marimo](https://marimo.io) for the [alphaXiv x marimo notebook competition](https://marimo.io/pages/events/notebook-competition).*

    *Paper: [Prompt Repetition Improves Non-Reasoning LLMs](https://arxiv.org/abs/2512.14982) (Leviathan, Kalman & Matias, 2025)*
    """)
    return


if __name__ == "__main__":
    app.run()

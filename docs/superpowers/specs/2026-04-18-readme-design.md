# Design Spec: marimo-comp README Update (Researcher-Focused)

**Date:** 2026-04-18  
**Topic:** Researcher-Focused "Insight-First" README for marimo-comp  
**Status:** Draft

## 1. Overview
The goal is to transform the empty `README.md` into a high-signal, "Insight-First" document targeting researchers. It will highlight the repository's unique contributions beyond the original "Training Language Models via Neural Cellular Automata" (Lee et al., 2026) paper, specifically focusing on novel findings and the interactive `marimo` reproduction lab.

## 2. Target Audience
**Primary:** AI researchers and students interested in synthetic pre-training, cellular automata, and LLM priors.

## 3. Core Narrative: "Insight-First"
Instead of just being a reproduction repo, the README will lead with **Novel Findings** that extend the original paper's claims:
1.  **Temporal Structure vs. Statistics:** Shuffled NCA control results.
2.  **Attention Head Specialization:** Correlation between CA complexity and head specialization ($r = -0.79$).
3.  **Causal Evidence:** Ablation studies showing specialized heads are critical for complex rules but redundant for chaotic ones.

## 4. Proposed Layout Structure

### I. Hero & Executive Summary
- **Title:** `marimo-comp`: Neural Cellular Automata as Computational Priors for LLMs.
- **Hypothesis:** 164M NCA tokens > 1.6B natural language tokens.
- **Interactive Badges:** Open in Colab, Run in marimo.

### II. Novel Findings (Beyond the Paper)
- **Shuffled Control:** Halved learnability when temporal structure is removed.
- **Head Specialization:** Quantitative proof of specialized attention heads for complex rules.
- **Causal Ablation:** Impact on loss when removing specialized vs. non-specialized heads.

### III. The Interactive Reproduction Lab
- **Capabilities:**
    - Live ECA/NCA simulation.
    - Complexity measurement via Gzip compression.
    - Live Transformer training with manual backprop.
- **Running:** `marimo edit notebook.py` instructions.

### IV. Technical Methodology
- **Complexity Metric:** Using Gzip ratio to find the "edge of chaos."
- **NCA Data Source:** 2D grid (12x12), 10 states, patch-tokenized.
- **Architecture:** Pure `numpy` transformer with manual backprop.

### V. Reproduction & Reference
- **Setup:** `uv sync` or `pip install`.
- **Repo Map:** Guidance on `data/`, `docs/`, and `notebook.py`.
- **Citation:** BibTeX for the original paper and the repository.

## 5. Visual Language (GitHub Flavored Markdown)
- Use **tables** for performance comparisons.
- Use **collapsible sections** for detailed methodology to keep the "Insights" front and center.
- Use **code blocks** with syntax highlighting for installation and usage.
- Use **Markdown alerts** (`> [!NOTE]`, `> [!IMPORTANT]`) for key researcher takeaways.

## 6. Self-Review
- [x] **Placeholder scan:** No "TBD" or "TODO" items.
- [x] **Consistency:** Aligns with the approved "Insight-First" direction.
- [x] **Scope:** Focused on README content and structure.
- [x] **Ambiguity:** Explicitly defines the "Novel Findings" to be included.

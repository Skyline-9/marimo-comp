# The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning

**Paper:** Sahraee-Ardakan, Delbracio, Milanfar (Google, 2026) — [arXiv:2602.18428](https://arxiv.org/abs/2602.18428)
**Competition pick: #3 (impressive math, harder to make visual)**

---

## 1. Core Finding

Autonomous (noise-agnostic) diffusion models learn a single time-invariant vector field that implicitly performs Riemannian gradient flow on a "Marginal Energy" landscape. They don't need to know the noise level because the geometry of the noisy data itself encodes that information.

## 2. The Key Concepts

### Marginal Energy
E_marg(u) = -log p(u), where p(u) is the marginal density of noisy data integrated over all noise levels.

### The Paradox
Near the data manifold, the gradient of this energy diverges (infinitely deep potential well). How does a bounded neural network remain stable?

### The Resolution
The learned vector field implicitly includes a local conformal metric (effective gain λ(t)) that perfectly counterbalances the geometric singularity. The field decomposes into:
1. Natural gradient of marginal energy (scaled by λ)
2. Transport correction (covariance term)
3. Linear drift

### Why "Blind" Models Can See
Two regimes where noise level is implicitly inferred:
- **High dimensions (D >> d):** The input u is a deterministic proxy for noise level
- **Near manifold:** Likelihood dominated by smallest noise scales → p(t|u) concentrates at t→0

## 3. Stability by Parameterization

| Parameterization | Stability | Why |
|-----------------|-----------|-----|
| Noise prediction (DDPM) | Unstable | High-gain amplifier for estimation errors |
| Signal prediction (EDM) | Stable | Error vanishes exponentially near data |
| Velocity prediction (Flow Matching) | Stable | Bounded gain, no singular coefficients |

## 4. Results

On CIFAR-10, SVHN, Fashion MNIST:
- DDPM Blind: fails catastrophically (artifacts, noise)
- Flow Matching Blind: sharp samples comparable to conditional version
- Confirms velocity parameterization is inherently stable for autonomous models

## 5. Why It's Tier 3 for Competition

- The core contribution is **theoretical** (Riemannian geometry proofs)
- Can demo on 2D toy data (Gaussian mixtures, Swiss roll) in numpy
- But the visual payoff per effort is lower than NCA
- Harder to make intuitive for non-math audiences
- The "aha moment" requires understanding of differential geometry

## 6. Extension Papers

| # | Paper | Key Idea |
|---|-------|----------|
| 1 | Energy Matching (2504.10612) | Unifies flow matching + energy-based models |
| 2 | Distance Marching (2602.02928) | Another time-unconditional approach |
| 3 | Riemannian Consistency Model (2510.00983) | Consistency models on manifolds |
| 4 | Riemannian AmbientFlow (2601.18728) | Joint manifold + density learning |
| 5 | Score Matching + Langevin (1907.05600) | Foundational score-based approach |
| 6 | Information Dynamics of Diffusion (2508.19897) | Info-theoretic view |
| 7 | Diffusion on Implicit Manifolds (2604.07213) | Diffusion without explicit manifold |
| 8 | Riemannian Metric for Diffusion (2510.05509) | Learn the geometry |
| 9 | MAD: Manifold Attracted Diffusion (2509.24710) | Noise-aware diffusion |
| 10 | Deep MMD Gradient Flow (2405.06780) | Gradient flow without adversarial training |

## 7. Possible WASM Demos (if chosen)

1. 2D toy diffusion: Gaussian mixture denoising with/without noise conditioning
2. Marginal energy landscape visualization (contour plots)
3. Vector field comparison: autonomous vs conditional
4. Stability demo: noise vs velocity parameterization failure/success
5. Interactive: user adds noise to 2D data, watches denoising flow

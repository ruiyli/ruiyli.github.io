---
layout: post
title: Diffusion Models Part II — DDPM = Autoregressive VAE
date: 2026-01-19 15:44:00
description: Diffusion Models Part II
tags: LearningNotes
categories: DiffusionModels
citation: true
tabs: true
---


In the article [Diffusion Models Part I — DDPM = Demolishing + Reconstructing](https://ruiyli.github.io/blog/2026/Diffusion-Models-Part-I/), we constructed a "demolish-reconstruct" analogy for the Denoising Diffusion Probabilistic Model (DDPM) and fully derived its theoretical formulation using this analogy. We also noted that DDPM is no longer a traditional diffusion model — it is more accurately viewed as a Variational Autoencoder (VAE). Indeed, the original DDPM paper also derives it from a VAE perspective.

Therefore, this article re-introduces DDPM from the VAE perspective and shares a Keras implementation code and practical experience.

---

## Multi-Step Breakthrough

In traditional VAEs, the encoding and generation processes are one-step:

$$
\begin{aligned}
&\textbf{Encoding}: x \rightarrow z \\
&\textbf{Generation}: z \rightarrow x 
\end{aligned} \tag{1}
$$

This involves only three distributions: the encoding distribution $$p(z \mid x)$$, the generation distribution $$q(x \mid z)$$, and the prior distribution $$q(z)$$. The advantage is simplicity — the mapping between $$x$$ and $$z$$ is deterministic, enabling both encoding and generation models, and supporting latent variable editing. However, the limitation is severe: since our ability to model probability distributions is limited, all three distributions are typically modeled as Gaussians, restricting expressive power and often yielding blurry generation results.

To overcome this, DDPM decomposes encoding and generation into $$T$$ steps:

$$
\begin{aligned}
&\textbf{Encoding}: \boldsymbol{x} = \boldsymbol{x}_0 \rightarrow \boldsymbol{x}_1 \rightarrow \boldsymbol{x}_2 \rightarrow \cdots \rightarrow \boldsymbol{x}_{T-1} \rightarrow \boldsymbol{x}_T = \boldsymbol{z} \\
&\textbf{Generation}: \boldsymbol{z} = \boldsymbol{x}_T \rightarrow \boldsymbol{x}_{T-1} \rightarrow \boldsymbol{x}_{T-2} \rightarrow \cdots \rightarrow \boldsymbol{x}_1 \rightarrow \boldsymbol{x}_0 = \boldsymbol{x}
\end{aligned} \tag{2}
$$

Each transition $$p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})$$ and $$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$$ models only a small change, still assumed Gaussian. One might ask: if both are Gaussian, why is multi-step better than single-step? The answer lies in approximation: small changes can be well-modeled by Gaussians, analogous to approximating a curve locally with a straight line. Multi-step decomposition resembles piecewise linear approximation of complex curves, theoretically overcoming the fitting limitations of single-step VAEs.

---

## Joint Divergence

Our plan is to enhance traditional VAEs via recursive decomposition (2). Each encoding step is modeled as $$p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})$$, and each generation step as $$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$$. The corresponding joint distributions are:

$$
\begin{aligned}
p(\boldsymbol{x}_0, \boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_T) &= p(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1}) \cdots p(\boldsymbol{x}_2 \mid \boldsymbol{x}_1) p(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \\
q(\boldsymbol{x}_0, \boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_T) &= q(\boldsymbol{x}_0 \mid \boldsymbol{x}_1) \cdots q(\boldsymbol{x}_{T-2} \mid \boldsymbol{x}_{T-1}) q(\boldsymbol{x}_{T-1} \mid \boldsymbol{x}_T) q(\boldsymbol{x}_T)
\end{aligned} \tag{3}
$$

Note: $$\boldsymbol{x}_0$$ represents real data, so $$\tilde{p}(\boldsymbol{x}_0)$$ is the data distribution; $$\boldsymbol{x}_T$$ is the final encoded representation, so $$q(\boldsymbol{x}_T)$$ is the prior; the rest $$p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})$$ and $$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$$ represent small encoding/generation steps.

(Note: We follow the notation convention used in this blog for VAEs — "encoding distribution uses $$p$$, generation distribution uses $$q$$" — which is opposite to the DDPM paper.)

As discussed in [Variational Autoencoder (Part 2): From a Bayesian Perspective](https://kexue.fm/archives/5343), the most concise theoretical understanding of VAEs is as minimizing the KL divergence between joint distributions. DDPM follows the same principle. We have written the two joint distributions above, so DDPM's objective is to minimize:

$$
KL(p \| q) = \int p(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1}) \cdots p(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log \frac{p(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1}) \cdots p(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0)}{q(\boldsymbol{x}_0 \mid \boldsymbol{x}_1) \cdots q(\boldsymbol{x}_{T-1} \mid \boldsymbol{x}_T) q(\boldsymbol{x}_T)} d\boldsymbol{x}_0 d\boldsymbol{x}_1 \cdots d\boldsymbol{x}_T \tag{4}
$$

This is DDPM's optimization target. So far, results align with the original DDPM paper (with minor notation differences) and the earlier work [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585). Next, we specify the forms of $$p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})$$ and $$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$$ and simplify the objective (4).

---

## Divide and Conquer

DDPM aims only to build a generative model, so it models each encoding step as a simple Gaussian: $$p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) = \mathcal{N}(\boldsymbol{x}_t; \alpha_t \boldsymbol{x}_{t-1}, \beta_t^2 \boldsymbol{I})$$. The mean is simply a scalar multiple of the input $$\boldsymbol{x}_{t-1}$$ — unlike traditional VAEs, where mean and variance are learned via neural networks. Thus, DDPM abandons encoding capability, yielding a pure generative model. The generation step $$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$$ is modeled as a Gaussian with learnable mean: $$\mathcal{N}(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}(\boldsymbol{x}_t), \sigma_t^2 \boldsymbol{I})$$. Here, $$\alpha_t, \beta_t, \sigma_t$$ are fixed (not trainable), set beforehand (we'll discuss how later); only $$\boldsymbol{\mu}(\boldsymbol{x}_t)$$ is trainable.

(Note: Our definitions of $$\alpha_t, \beta_t$$ differ from the original paper.)

Since $$p$$ contains no trainable parameters, the integral over $$p$$ in (4) contributes only a negligible constant. Thus, (4) is equivalent to:

$$
\begin{aligned}
&-\int p(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1}) \cdots p(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_0 \mid \boldsymbol{x}_1) \cdots q(\boldsymbol{x}_{T-1} \mid \boldsymbol{x}_T) q(\boldsymbol{x}_T) d\boldsymbol{x}_0 d\boldsymbol{x}_1 \cdots d\boldsymbol{x}_T \\
= &-\int p(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1}) \cdots p(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \left[ \log q(\boldsymbol{x}_T) + \sum_{t=1}^{T} \log q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) \right] d\boldsymbol{x}_0 d\boldsymbol{x}_1 \cdots d\boldsymbol{x}_T
\end{aligned} \tag{5}
$$

Since the prior $$q(\boldsymbol{x}_T)$$ is typically standard Gaussian (no parameters), this term contributes only a constant. Thus, we need to compute each term:

$$
\begin{aligned}
&-\int p(\boldsymbol{x}_T \mid \boldsymbol{x}_{t-1}) \cdots p(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) d\boldsymbol{x}_0 d\boldsymbol{x}_1 \cdots d\boldsymbol{x}_T \\
= &-\int p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) \cdots p(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) d\boldsymbol{x}_0 d\boldsymbol{x}_1 \cdots d\boldsymbol{x}_t \\
= &-\int p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) p(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) d\boldsymbol{x}_0 d\boldsymbol{x}_{t-1} d\boldsymbol{x}_t
\end{aligned} \tag{6}
$$

The first equality holds because $$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$$ depends at most on $$\boldsymbol{x}_t$$, so integrals over $$\boldsymbol{x}_{t+1}$$ to $$\boldsymbol{x}_T$$ yield 1. The second equality holds because $$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$$ does not depend on $$\boldsymbol{x}_1, \cdots, \boldsymbol{x}_{t-2}$$, so their integrals can be precomputed, yielding $$p(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t-1}; \bar{\alpha}_{t-1} \boldsymbol{x}_0, \bar{\beta}_{t-1}^2 \boldsymbol{I})$$, as shown in the next section (Eq. 9).

---

## Scene Reconstruction

The following steps mirror the "How to Reconstruct" section in the [previous article](https://ruiyli.github.io/blog/2026/Diffusion-Models-Part-I/):

1. Ignoring optimization-irrelevant constants, $$-\log q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$$ contributes $$\frac{1}{2\sigma_t^2} \|\boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\|^2$$.

2. $$p(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0)$$ implies $$\boldsymbol{x}_{t-1} = \bar{\alpha}_{t-1} \boldsymbol{x}_0 + \bar{\beta}_{t-1} \bar{\boldsymbol{\varepsilon}}_{t-1}$$; $$p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})$$ implies $$\boldsymbol{x}_t = \alpha_t \boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t$$, where $$\bar{\boldsymbol{\varepsilon}}_{t-1}$$ and $$\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$$.

3. From $$\boldsymbol{x}_{t-1} = \frac{1}{\alpha_t} (\boldsymbol{x}_t - \beta_t \boldsymbol{\varepsilon}_t)$$, we parameterize $$\boldsymbol{\mu}(\boldsymbol{x}_t) = \frac{1}{\alpha_t} (\boldsymbol{x}_t - \beta_t \boldsymbol{\varepsilon}_\theta(\boldsymbol{x}_t, t))$$.

This transforms the optimization target into:

$$
\frac{\beta_t^2}{\alpha_t^2 \sigma_t^2} \mathbb{E}_{\bar{\varepsilon}_{t-1}, \varepsilon_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I}), \boldsymbol{x}_0 \sim \tilde{p}(\boldsymbol{x}_0)} \left[ \left\| \boldsymbol{\varepsilon}_t - \boldsymbol{\varepsilon}_\theta(\bar{\alpha}_t \boldsymbol{x}_0 + \alpha_t \bar{\beta}_{t-1} \bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t, t) \right\|^2 \right] \tag{7}
$$

After applying the "variance reduction" substitution from the [previous article](https://ruiyli.github.io/blog/2026/Diffusion-Models-Part-I/), we get:

$$
\frac{\beta_t^4}{\bar{\beta}_t^2 \alpha_t^2 \sigma_t^2} \mathbb{E}_{\varepsilon \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I}), \boldsymbol{x}_0 \sim \tilde{p}(\boldsymbol{x}_0)} \left[ \left\| \boldsymbol{\varepsilon} - \frac{\bar{\beta}_t}{\beta_t} \boldsymbol{\varepsilon}_\theta(\bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}, t) \right\|^2 \right] \tag{8}
$$

This is DDPM's training objective (the original paper found that removing the leading coefficient improves performance). We derived this step-by-step from the VAE objective — lengthy, but logically sound with no conceptual leaps.

In contrast, the original DDPM paper abruptly introduces $$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0)$$ (their notation) to cancel terms and convert to KL divergence of Gaussians — a highly technical trick that feels "mysterious" and hard to accept intuitively.

---

## Hyperparameter Settings

This section discusses choices for $$\alpha_t, \beta_t, \sigma_t$$.

For $$p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})$$, it's conventional to set $$\alpha_t^2 + \beta_t^2 = 1$$, halving the parameters and simplifying forms. As derived previously, due to Gaussian additivity under this constraint:

$$
p(\boldsymbol{x}_t \mid \boldsymbol{x}_0) = \int p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) \cdots p(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) d\boldsymbol{x}_1 \cdots d\boldsymbol{x}_{t-1} = \mathcal{N}(\boldsymbol{x}_t; \bar{\alpha}_t \boldsymbol{x}_0, \bar{\beta}_t^2 \boldsymbol{I}) \tag{9}
$$

where $$\bar{\alpha}_t = \alpha_1 \cdots \alpha_t$$ and $$\bar{\beta}_t = \sqrt{1 - \bar{\alpha}_t^2}$$. This yields a simple form for $$p(\boldsymbol{x}_t \mid \boldsymbol{x}_0)$$. 

Why choose $$\alpha_t^2 + \beta_t^2 = 1$$? Since $$p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) = \mathcal{N}(\boldsymbol{x}_t; \alpha_t \boldsymbol{x}_{t-1}, \beta_t^2 \boldsymbol{I})$$ implies $$\boldsymbol{x}_t = \alpha_t \boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t$$, where $$\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$$, if $$\boldsymbol{x}_{t-1} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$$, we want $$\boldsymbol{x}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$$, hence $$\alpha_t^2 + \beta_t^2 = 1$$.

We typically set $$q(\boldsymbol{x}_T) = \mathcal{N}(\boldsymbol{x}_T; \mathbf{0}, \boldsymbol{I})$$. Since we minimize KL divergence between joint distributions (i.e., $$p = q$$), their marginals should also match. Thus, we want:

$$
q(\boldsymbol{x}_T) = \int p(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1}) \cdots p(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) d\boldsymbol{x}_0 d\boldsymbol{x}_1 \cdots d\boldsymbol{x}_{T-1} = \int p(\boldsymbol{x}_T \mid \boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) d\boldsymbol{x}_0 \tag{10}
$$

Since $$\tilde{p}(\boldsymbol{x}_0)$$ is arbitrary, the only way for (10) to hold is if $$p(\boldsymbol{x}_T \mid \boldsymbol{x}_0) = q(\boldsymbol{x}_T)$$ — i.e., independent of $$\boldsymbol{x}_0$$, meaning $$\bar{\alpha}_T \approx 0$$. This confirms DDPM has no encoding capability — $$p(\boldsymbol{x}_T \mid \boldsymbol{x}_0)$$ is effectively independent of input $$\boldsymbol{x}_0$$. In the "demolish-reconstruct" analogy: the original building is fully dismantled into raw materials; rebuilding can yield any structure, not necessarily the original. DDPM uses $$\alpha_t = \sqrt{1 - \frac{0.02t}{T}}$$; its properties were analyzed in the [previous article](https://ruiyli.github.io/blog/2026/Diffusion-Models-Part-I/)'s "Hyperparameter Settings" section.

For $$\sigma_t$$, theoretically, optimal values depend on $$\tilde{p}(\boldsymbol{x}_0)$$, but we avoid making $$\sigma_t$$ trainable. Instead, we derive optimal $$\sigma_t$$ for special cases of $$\tilde{p}(\boldsymbol{x}_0)$$ and assume generalization:

1. If training set has one sample $$\boldsymbol{x}_*$$ (i.e., $$\tilde{p}(\boldsymbol{x}_0) = \delta(\boldsymbol{x}_0 - \boldsymbol{x}_*)$$), optimal $$\sigma_t = \frac{\bar{\beta}_{t-1}}{\bar{\beta}_t} \beta_t$$.
2. If $$\tilde{p}(\boldsymbol{x}_0) \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$$, optimal $$\sigma_t = \beta_t$$.

Experiments show similar performance; either can be used. Derivations are lengthy and deferred.

---

## Reference Implementation

**GitHub**: [https://github.com/bojone/Keras-DDPM](https://github.com/bojone/Keras-DDPM)

Note: The implementation does not strictly follow the original DDPM code. It simplified the U-Net architecture (e.g., replacing feature concatenation with addition, removing Attention) for faster results. Tested on a single 24GB RTX 3090, training 128×128 CelebA HQ with `blocks=1`, `batch_size=64`, results appear within half a day. Sampling after 3 days:

<div align="center">
  {% include figure.liquid path="assets/img/20260119/ddpm_sample.png" class="img-fluid rounded z-depth-1" width="800" alt="ddpm_sample" %}
  <p style="text-align: center; margin-top: 2px; font-size: 16px">DDPM Sampling Results</p>
</div>

**Practical Tips from Debugging**:

1. **Loss Function**: Use squared Euclidean distance, not MSE. MSE divides by image size (width × height × channels), making loss too small and gradients vanish, causing convergence followed by divergence (common in low-precision training; see [Using Mixed Precision and XLA in bert4keras](https://kexue.fm/archives/9059)).
2. **Normalization**: Use Instance Norm, Layer Norm, or Group Norm — avoid Batch Norm due to train-inference inconsistency (may yield great training but poor inference).
3. **Architecture**: No need to copy the original paper's U-Net exactly. Follow U-Net's encoder-decoder structure — it's essentially a regression problem, easy to train.
4. **Time Embedding**: Original paper uses sinusoidal positional encoding; I found trainable Embedding works similarly well.
5. **Optimizer**: I used LAMB (from language model pretraining habits) — easier to tune learning rate; $$10^{-3}$$ works well for most initializations.

---

## Comprehensive Evaluation

Combining [Diffusion Models Part I — DDPM = Demolishing + Reconstructing](https://ruiyli.github.io/blog/2026/Diffusion-Models-Part-I/) and this article, readers should now have a clear view of DDPM's strengths, weaknesses, and improvement directions.

**Strengths**: Easy to train, generates sharp images. Compared to GANs (min-max, unstable, prone to collapse), DDPM is a pure regression loss — training is stable. The "demolish-reconstruct" analogy also makes DDPM intuitively accessible.

**Weaknesses**:
1. **Slow Sampling**: Requires $$T$$ steps (original paper: $$T=1000$$), $$T$$ times slower than GAN's one-step sampling. Many works address this.
2. **No Latent Control**: GANs map noise deterministically to output — enabling interpolation and editing. DDPM's generation is stochastic — no deterministic mapping, so editing is impossible. The paper's "interpolation" is just interpolating original images, then adding noise for the model to "fill in" — hard to achieve semantic fusion.

**Other Directions**:
- **Conditional DDPM**: Like C-VAE/C-GAN, conditional generation (e.g., text-to-image, super-resolution) is a major application (e.g., Google’s Imagen).
- **Discrete Data**: DDPM is designed for continuous variables, but its ideas may extend to discrete data — how to design DDPM for discrete data?

---

## Related Work

Most associate DDPM with traditional diffusion models, energy models, or denoising autoencoders. But I want to highlight a prior work from this blog: [Powerful NVAE: No More Blurry VAE Images](https://kexue.fm/archives/7574). From a VAE perspective, traditional VAEs generate blurry images; DDPM is (to my knowledge) the second VAE capable of sharp images — the first being NVAE.

NVAE introduces many latent variables $$z = \{z_1, z_2, \cdots, z_L\}$$ with recursive relationships — its sampling process resembles DDPM's. Theoretically, DDPM is a highly simplified NVAE: latent variables modeled as Markovian conditional Gaussians (vs. NVAE's non-Markovian), and generation via iterative reuse of one model (vs. NVAE's large model using all $$z = \{z_1, z_2, \cdots, z_L\}$$). But NVAE's parameter sharing is conceptually similar to model reuse.

---

## Summary

This article derived DDPM from a VAE perspective, revealing it as a simplified autoregressive VAE, closely related to NVAE. We shared implementation code, practical tips, and a comprehensive evaluation of DDPM.


**References**:  
- [https://kexue.fm/archives/9152](https://kexue.fm/archives/9152)

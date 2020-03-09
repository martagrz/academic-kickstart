---
title: 'Introduction to Variational Inference'
date: 2020-03-09
markup: mmark
tags: 
- Variational Inference
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
  
---

# Introduction

Monte Carlo Markov Chain (MCMC) methods are established as ways to approximate difficult to calculate or intractable probability densities. Variational inference (VI) is a method from machine learning that also approximates probability densities but through optimisation rather than sampling from a Markov Chain. It two main steps: first, a family of densities is specified which is believed to represent the closest approximation to the true density and secondly, a member of that family is chosen which is closest to the target distribution. *Closeness* uses metrics of difference between probability distributions and KL-divergence is the most commonly used. 

The goal of these methods is the same and it is to approximate a conditional density of latent variables given observed variables. Specifically, given a joint distribution $p(z, x)$ of observed $x$ and latent $z$, we seek to approximate $p(z|x)$. The conditional distribution is useful as it can be used to, for example, compute point or interval estimates of latent variables or form predictive densities of new data. 

$$p(z|x) = \frac{p(z, x)}{p(x)} = \frac{p(z, x)}{\int p(z, x) \text{d}z}$$

We can use the joint distribution to calculate the sought-after conditional, and the above equation shows how that can be done. It is unlikely that $p(x)$, the density of the observed data, is known and it is replaced by marginalising out the latent variable from the joint distribution. However, this integral, also known as the **evidence** is often intractable or difficult to calculate. Therefore, MCMC or VI methods are used to approximate this conditional distribution.

# Method

A family $Q$ of densities over the latent variables is specified, where $q(z) \in Q$ is considered as the approximation to the exact distribution

$$q^*(z) = \arg \min_{q(z) \in Q} \text{KL} (q(z) \parallel p(z|x))$$ 

where, $$\text{KL}(q(z)\parallel p(z|x))=q(z)\log \left({\frac {q(z)}{p(z|x)}}\right)\,\text{d}z$$ is the Kullback-Leiber divergence. It is important to note that this measure is **asymmetric**, meaning $$\text{KL} (q(z) \parallel p(z|x)) \neq \text{KL} (p(z|x) \parallel q(z))$$. It is non-negative measure, and a measure of $0$ indicated that the two distributions are identical.

The complexity of the family determines the complexity of the computation of this optimisation. However, the above optimisation is not computable due to the reliance of $\log p(x)$.

$$ \text{KL}(q(z)\parallel p(z|x)) = \mathbb{E} \left[ \log q(z) \right] - \mathbb{E} \left[ \log p(z|x) \right] $$ 

$$ \text{KL}(q(z)\parallel p(z|x)) = \mathbb{E} \left[ \log q(z) \right] - \mathbb{E} \left[ \log p(z, x) \right] + \log p(x) $$ 

since the expectation is taken with respect to $q(z)$. We cannot compute this, so we optimisise an alternative that is equivalent to KL up to a constant 

$$\text{ELBO} (q) = \mathbb{E} \left[ \log p(z,x) \right] - \mathbb{E} \left[ \log q(z) \right] $$

Therefore, $$\text{ELBO}(q) = - \text{KL} + \log p(x)$$ and maximising the ELBO is equivalent to minimising the KL divergence between the two distributions. The ELBO can be described as the sum of expected log likelihood of data and the KL divergence between the prior $p(z)$ and $q(z)$. The first term encourages densities that place their mass on configurations of latent variables that explain the observed data, while the second term encourages densities close to the prior. As found often in Bayesian statistics, it is a balance between the likelihood and the prior. 

The ELBO stands for the **evidence lower-bound** and it is exactly that: 

$$ \log p(x) \geq \text{ELBO} (q) \text{ for any } q(z)$$

since

$$ \log p(x) = \text{KL} (q(z) \parallel p(z|x)) + \text{ELBO}(q) $$ 

and $$\text{KL}(\cdot) \geq 0$$ by construct of the measure, or it can be derived by Jensen's inequality, which we omit here. This bound is a good approximation of marginal likelihood, and can sometimes provide a basis for model selection, although this is not supported in theory. 

## Distributional family $Q$

An example of $Q$ that is often chosen is the mean-field, where latent variables are mutually independent and each governed by a distinct factor in the variational density. It connects the fitted variational density to data and model. 

$$ q(z) = \Pi^m_{j=1} q_j(z_j) $$

### Example 

This example is taken from Blei et al (2017). It compares the exact 2-dimensional Gaussian posterior and the variational approximation. 

![png](/img/posts/vi/mean-field-example.png)

The variational approximation has the same mean as the original density but the covariance, by construction, is decoupled. This is a direct result of the mean-field approximation, as we have assumed each $q_j(z_j)$ to be independent. The marginal variances of approximation under-represent that of the target density. 

Moreover, the KL divergence is not symmetric, meaning that it penalises placing mass in $q(\cdot)$ on areas where $p(\cdot)$ has little mass but penalises less the reverse. In the above example, $q(\cdot)$ would have to expand into territory where $p(\cdot)$ has little mass.


## Optimisation method 

A common method of optimisation under the mean-field assumption is the **coordinate ascent mean-field variational inference** (CAVI) (Bishop, 2006) which iteratively optimises each factor of the mean-field variational density, holding the others fixed. It obtains the ELBO local optimum. 

$$ q^*_{j} (z_j) \propto \exp \{ \mathbb{E}_{-j} [  \log p(z_j | z_{-j}, x)  ]  \}  \equiv \exp \{ \mathbb{E}_{-j} [  \log p(z_j, z_{-j}, x)  ]  \} $$

The equivalence holds under the mean-field assumption since the latent variables are assumed to be independent. The algorithm calculates the above for each latent variable $$j \in \{ 1,\dots, m \}$$ and then computes $$\text{ELBO} (q) = \mathbb{E} \left[ \log p(z,x) \right] - \mathbb{E} \left[ \log q(z) \right] $$ until ELBO has converged (differences in each iteration are negligible).

This algorithm is closely related to Gibbs sampling, which maintains a realisation of latent variables and iteratively samples from each variable's complete conditional distribution. 



# References
* Bishop, C.M., 2006. Pattern recognition and machine learning. springer.
* Blei, D.M., Kucukelbir, A. and McAuliffe, J.D., 2017. Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518), pp.859-877.